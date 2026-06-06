import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from reward import RewardConfig, RewardCalculator
from reshuffle import MinMaxReshuffler

@dataclass
class DepotConfig:
    n_blocks: int = 6
    n_bays:   int = 4
    n_rows:   int = 4
    n_tiers:  int = 6
    n_ticks:  int = 30
    max_in:   int = 5
    max_out:  int = 5
    # Fraction of yard capacity (0.0–1.0) to pre-fill at reset, spread flat across stacks.
    # Applied in both training and evaluation (snapshotted in config_used.ini) so the
    # regime is identical for fairness. 0.0 = start empty.
    warmstart: float = 0.0


class DepotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: DepotConfig = None, reward_config: RewardConfig = None):
        super().__init__()
        self.config = config or DepotConfig()
        c = self.config

        # Shorthand
        self.B, self.Ba, self.R, self.H = c.n_blocks, c.n_bays, c.n_rows, c.n_tiers

        # Action space sizes
        self.n_inbound  = self.B * self.Ba * self.R
        self.n_outbound = self.B * self.Ba * self.R * self.H
        self.n_actions  = self.n_inbound + self.n_outbound

        # Observation feature count. Two per-tier channels (occupancy + dwell) keep full
        # grid information without a -1 sentinel; four per-stack summaries (height,
        # top_dwell, max_dwell, dist_to_cr) expose the decision-relevant order statistics
        # directly so the policy needn't recompute them from the raw grid. See _get_obs.
        grid_size    = self.B * self.Ba * self.R * self.H   # per channel
        stack_size   = self.B * self.Ba * self.R            # per per-stack feature
        n_grid_ch    = 2   # occupancy, dwell
        n_stack_feat = 4   # height, top_dwell, max_dwell, dist_to_cr
        max_d_size   = self.B
        scalar_size  = 8   # mode, tick, crane_pos(3), occ, mean_d, std_d
        self.obs_size = (n_grid_ch * grid_size
                         + n_stack_feat * stack_size
                         + max_d_size + scalar_size)

        # Gymnasium spaces. Every feature is now >= 0 (empty slots encode as 0 in the
        # dwell channel and are flagged by the occupancy channel), so low=0.
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        # Internal env state (initialised properly in reset)
        self.grid = None
        self.mode = 0
        self.tick = 0
        self.n_in_remaining  = 0
        self.n_out_remaining = 0
        self.crane_pos = np.array([0, 0, 0], dtype=np.int32)

        self.reward_calc = RewardCalculator(reward_config or RewardConfig(), self)
        self.reshuffler  = MinMaxReshuffler(self)
        self._last_breakdown = {}  # for logging

    def _get_obs(self) -> np.ndarray:
        """# Build flat observation vector.

        ## Grid channels (B, Ba, R, H):
        - occupancy: 1 if slot occupied, 0 if empty
        - dwell: how long the container in the slot has been there; 0 for empty slots (because of the occupancy channel, no -1 sentinel needed)
        
        ## Per-stack summaries (B, Ba, R):
        - height: how many tiers are occupied
        - top_dwell: dwell of the topmost container (0 for empty stacks)
        - max_dwell: dwell of the oldest container (0 for empty stacks)
        - dist_to_cr: Manhattan distance from the stack to the crane's current position
        """
        occupied_mask = self.grid >= 0                       # (B, Ba, R, H)
        occupancy = occupied_mask.astype(np.float32)         # 1 occupied / 0 empty
        dwell_ch = np.where(occupied_mask, self.grid, 0).astype(np.float32)

        heights = occupied_mask.sum(axis=3)                  # (B, Ba, R)

        # Topmost container's dwell: the 0-reshuffle retrieval option; 0 for empty stacks.
        top_idx = np.clip(heights - 1, 0, self.H - 1)
        top_dwell = np.take_along_axis(self.grid, top_idx[..., None], axis=3)[..., 0]
        top_dwell = np.where(heights > 0, top_dwell, 0).astype(np.float32)

        # Oldest container anywhere in the stack; 0 for empty stacks (-1 max otherwise).
        stack_max = self.grid.max(axis=3)
        max_dwell = np.where(stack_max >= 0, stack_max, 0).astype(np.float32)

        # Per-stack Manhattan distance from current crane position
        bl_idx = np.arange(self.B).reshape(self.B, 1, 1)
        ba_idx = np.arange(self.Ba).reshape(1, self.Ba, 1)
        r_idx  = np.arange(self.R).reshape(1, 1, self.R)
        dist_to_cr = (
            np.abs(bl_idx - self.crane_pos[0])
            + np.abs(ba_idx - self.crane_pos[1])
            + np.abs(r_idx - self.crane_pos[2])
        ).astype(np.float32)  # (B, Ba, R)

        occ_ratio = occupied_mask.sum() / self.grid.size

        # Yard-wide dwell stats over occupied slots (0 when yard empty)
        occupied_dwell = self.grid[occupied_mask]
        if occupied_dwell.size > 0:
            mean_d = float(occupied_dwell.mean())
            std_d = float(occupied_dwell.std())
        else:
            mean_d = 0.0
            std_d = 0.0

        # Per-block max dwell: 0 for an empty block (no -1 sentinel in the observation)
        block_max = self.grid.reshape(self.B, -1).max(axis=1)
        max_d = np.where(block_max >= 0, block_max, 0).astype(np.float32)

        obs = np.concatenate([
            occupancy.flatten(),
            dwell_ch.flatten(),
            heights.astype(np.float32).flatten(),
            top_dwell.flatten(),
            max_dwell.flatten(),
            dist_to_cr.flatten(),
            [float(self.mode)],
            [float(self.tick)],
            self.crane_pos.astype(np.float32),
            [occ_ratio],
            [mean_d],
            [std_d],
            max_d,
        ]).astype(np.float32)

        return obs

    def action_masks(self) -> np.ndarray:
        """
        Build boolean mask of length n_actions.
        Inbound mode  → only inbound indices valid (and only stacks with free tiers)
        Outbound mode → only outbound indices valid (and only occupied slots)
        """
        mask = np.zeros(self.n_actions, dtype=bool)

        if self.mode == 0:  # inbound
            # stack has free tier if any tier in the stack is -1
            occupied_per_stack = (self.grid >= 0).sum(axis=3)  # (B, Ba, R)
            valid_stacks = occupied_per_stack < self.H
            mask[:self.n_inbound] = valid_stacks.flatten()
        else:  # outbound
            # A buried slot is retrievable only if the containers above it can
            # actually be relocated: it needs (height-1-h) free tiers OUTSIDE its
            # own stack (the reshuffler can't use the source stack). Top containers
            # need 0 reshuffles, so they are always retrievable.
            heights = (self.grid >= 0).sum(axis=3)             # (B, Ba, R)
            total_free = self.grid.size - int((self.grid >= 0).sum())
            free_outside = total_free - (self.H - heights)     # (B, Ba, R)
            h_idx = np.arange(self.H).reshape(1, 1, 1, self.H)
            n_reshuffles = heights[..., None] - 1 - h_idx      # (B, Ba, R, H)
            feasible = (self.grid >= 0) & (free_outside[..., None] >= n_reshuffles)
            mask[self.n_inbound:] = feasible.flatten()

        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.full(
            (self.B, self.Ba, self.R, self.H),
            -1,
            dtype=np.int32,
        )
        self.tick = 0
        self.crane_pos = np.array([0, 0, 0], dtype=np.int32)

        self._warmstart_yard()  # optionally pre-fill the yard (config.warmstart)

        # Draw the first actionable tick's demand, skipping ticks with nothing to do
        while self.tick < self.config.n_ticks:
            self._draw_new_tick_demand()
            if self._resolve_mode():
                break
            self.tick += 1

        return self._get_obs(), {}

    def _warmstart_yard(self):
        """Pre-fill the yard flat to config.warmstart of capacity (0.0–1.0).

        "Flat" = containers spread as evenly as possible across stacks (heights differ by
        at most 1, the remainder placed on a random subset of stacks), filling bottom-up.
        Pre-placed containers get random dwell ages in [0, n_ticks) so the staged yard
        isn't a single uniform age. Uses self.np_random, so a given reset seed reproduces
        the same warmstart for every policy (paired-fair in evaluation).
        """
        frac = min(1.0, max(0.0, float(self.config.warmstart)))
        if frac <= 0.0:
            return
        capacity = self.grid.size
        n_fill = min(capacity, int(round(frac * capacity)))
        n_stacks = self.B * self.Ba * self.R

        base, extra = divmod(n_fill, n_stacks)
        heights = np.full(n_stacks, base, dtype=np.int32)
        if extra:
            heights[self.np_random.permutation(n_stacks)[:extra]] += 1
        heights = heights.reshape(self.B, self.Ba, self.R)

        tier = np.arange(self.H).reshape(1, 1, 1, self.H)
        occupied = tier < heights[..., None]                       # (B, Ba, R, H)
        # ages = self.np_random.integers(0, max(1, self.config.n_ticks), size=self.grid.shape)
        ages = np.zeros(self.grid.shape, dtype=np.int32)  # start warmstart containers with 0 dwell for a cleaner initial state
        self.grid = np.where(occupied, ages, -1).astype(np.int32)

    def step(self, action: int):
        decoded = self.decode_action(action)
        tick_at_step_start = self.tick

        if decoded[0] == "inbound":
            _, bl, ba, r = decoded
            self._apply_inbound(bl, ba, r)
            reward, breakdown = self.reward_calc.compute_inbound_step(bl, ba, r)
        else:
            _, bl, ba, r, h = decoded
            # Read the retrieved dwell BEFORE removal (the slot becomes -1 after)
            retrieved_dwell = int(self.grid[bl, ba, r, h])
            dist, reshuffle_dists = self._apply_outbound(bl, ba, r, h)
            reward, breakdown = self.reward_calc.compute_outbound_step(
                retrieved_dwell, dist, reshuffle_dists
            )

        self._update_mode_or_advance()

        # Per-tick penalty: applied if a tick advanced during this step
        if self.tick > tick_at_step_start:
            tick_pen = self.reward_calc.tick_penalty()
            reward += tick_pen
            breakdown["tick_penalty"] = tick_pen

        terminated = self.tick >= self.config.n_ticks
        truncated = False

        # Episode bonus at termination
        if terminated:
            ep_bonus = self.reward_calc.episode_bonus()
            reward += ep_bonus
            breakdown["episode_bonus"] = ep_bonus

        self._last_breakdown = breakdown
        info = {"reward_breakdown": breakdown}

        return self._get_obs(), float(reward), terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass


    # ─── Action encoding ────────────────────────────────────────────

    def encode_inbound(self, bl: int, ba: int, r: int) -> int:
        """(bl, ba, r) → flat action index in [0, n_inbound)."""
        return bl * (self.Ba * self.R) + ba * self.R + r

    def encode_outbound(self, bl: int, ba: int, r: int, h: int) -> int:
        """(bl, ba, r, h) → flat action index in [n_inbound, n_actions)."""
        return (self.n_inbound
                + bl * (self.Ba * self.R * self.H)
                + ba * (self.R * self.H)
                + r  * self.H
                + h)

    def decode_action(self, action: int):
        """
        Flat action index → tuple describing the action.
        Returns:
            ("inbound",  bl, ba, r)       if inbound
            ("outbound", bl, ba, r, h)    if outbound
        """
        if action < self.n_inbound:
            bl = action // (self.Ba * self.R)
            rem = action % (self.Ba * self.R)
            ba = rem // self.R
            r  = rem % self.R
            return ("inbound", bl, ba, r)
        else:
            local = action - self.n_inbound
            bl = local // (self.Ba * self.R * self.H)
            rem = local % (self.Ba * self.R * self.H)
            ba = rem // (self.R * self.H)
            rem = rem % (self.R * self.H)
            r  = rem // self.H
            h  = rem % self.H
            return ("outbound", bl, ba, r, h)
    
    # ─── Utilities ──────────────────────────────────────────────────

    def _stack_height(self, bl: int, ba: int, r: int) -> int:
        """Number of occupied tiers in a stack."""
        return int((self.grid[bl, ba, r, :] >= 0).sum())

    def _yard_occupancy(self) -> int:
        """Total occupied slots across the yard."""
        return int((self.grid >= 0).sum())

    def _yard_full(self) -> bool:
        """True when every slot in the yard is occupied."""
        return self._yard_occupancy() >= self.grid.size

    def _draw_new_tick_demand(self):
        """Sample inbound/outbound counts for a new tick."""
        self.n_in_remaining  = int(self.np_random.integers(0, self.config.max_in + 1))
        self.n_out_remaining = int(self.np_random.integers(0, self.config.max_out + 1))

    def _resolve_mode(self) -> bool:
        """
        Drop demand the current yard can't serve, then pick the starting mode
        for the tick. Returns False when nothing is actionable (caller skips
        the tick).

        - outbound demand is impossible from an empty yard  -> dropped
        - inbound demand is impossible into a full yard      -> dropped (full-drop edge)
        - outbound first when the yard can cover it, or when no inbound demand remains;
          otherwise inbound
        """
        if self.n_out_remaining > 0 and self._yard_occupancy() == 0:
            self.n_out_remaining = 0
        if self.n_in_remaining > 0 and self._yard_full():
            self.n_in_remaining = 0

        if self.n_in_remaining == 0 and self.n_out_remaining == 0:
            return False

        if self.n_out_remaining > 0 and (
            self._yard_occupancy() >= self.n_out_remaining or self.n_in_remaining == 0
        ):
            self.mode = 1  # outbound
        else:
            self.mode = 0  # inbound
        return True

    # ─── Transitions ────────────────────────────────────────────────

    def _apply_inbound(self, bl: int, ba: int, r: int):
        """Place a container at the next free tier of (bl, ba, r)."""
        h = self._stack_height(bl, ba, r)
        assert h < self.H, f"Stack ({bl},{ba},{r}) is full"
        self.grid[bl, ba, r, h] = 0  # dwell starts at 0
        self.n_in_remaining -= 1

    def _apply_outbound(self, bl: int, ba: int, r: int, h: int):
        """
        Retrieve container at (bl, ba, r, h).
        Relocates containers above target via MinMax.
        Returns (crane Manhattan distance, per-relocation block distances).
        """
        assert self.grid[bl, ba, r, h] >= 0, f"Slot ({bl},{ba},{r},{h}) is empty"

        # Crane travel BEFORE reshuffle moves
        prev_pos = self.crane_pos.copy()
        new_pos = np.array([bl, ba, r], dtype=np.int32)
        dist = int(np.abs(prev_pos - new_pos).sum())
        self.crane_pos = new_pos

        # Relocate containers above the target
        reshuffle_dists = self.reshuffler.relocate_above(bl, ba, r, h)

        # Remove the target container
        self.grid[bl, ba, r, h] = -1

        self.n_out_remaining -= 1
        return dist, reshuffle_dists

    def _advance_tick(self):
        """End-of-tick: age containers, increment tick, draw the next actionable demand."""
        self.grid[self.grid >= 0] += 1
        self.tick += 1

        # Skip ticks with nothing actionable (empty draws, or demand the yard can't serve)
        while self.tick < self.config.n_ticks:
            self._draw_new_tick_demand()
            if self._resolve_mode():
                return
            self.grid[self.grid >= 0] += 1
            self.tick += 1

    def _update_mode_or_advance(self):
        """
        After an action, keep draining the current mode while it has demand AND
        remains feasible; otherwise drop unservable demand, then switch mode or
        advance the tick.
        """
        if self.mode == 0:  # finished an inbound
            if self.n_in_remaining > 0 and not self._yard_full():
                return  # keep placing
            self.n_in_remaining = 0  # drop inbound that no longer fits (full-drop edge)
            if self.n_out_remaining > 0 and self._yard_occupancy() > 0:
                self.mode = 1
            else:
                self.n_out_remaining = 0
                self._advance_tick()
        else:  # finished an outbound
            if self.n_out_remaining > 0 and self._yard_occupancy() > 0:
                return  # keep retrieving
            self.n_out_remaining = 0  # drop outbound demand the empty yard can't serve
            if self.n_in_remaining > 0 and not self._yard_full():
                self.mode = 0
            else:
                self.n_in_remaining = 0
                self._advance_tick()