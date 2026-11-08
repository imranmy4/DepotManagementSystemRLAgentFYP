"""Order-at-a-time yard operation — the logic the deployment API wraps.

An order (inbound / outbound) arrives, the policy picks an action, and the action is turned
into an ordered list of crane instructions: for an outbound, the MinMax reshuffler is
dry-run on a copy of the grid to produce each relocation, then the retrieve. Steps are
applied one at a time (Next) or all at once (Complete). No UI code here.

Moved unchanged in behaviour from the legacy Streamlit ops dashboard.
"""
import dataclasses

import numpy as np

from depot.env import DepotEnv


class LiveOrderRun:
    """
    Manages a DepotEnv in order-at-a-time mode.

    step_idx = index of the NEXT UNAPPLIED step in self.steps.
      -1          → no active order yet
      0..N-1      → current_step = steps[step_idx] (shown, not applied)
      >= N        → order complete
    """

    def __init__(self, depot_cfg, reward_cfg, policy, warmstart=0.4, seed=42):
        self._depot_cfg = depot_cfg
        self._reward_cfg = reward_cfg
        self.policy = policy          # anything with .act(env) -> int (depot.inference.Policy)
        self._warmstart = warmstart
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        ws_cfg = dataclasses.replace(self._depot_cfg, warmstart=self._warmstart)
        self.env = DepotEnv(ws_cfg, reward_config=self._reward_cfg)
        self.env.reset(seed=0)
        self.env.n_in_remaining = 0
        self.env.n_out_remaining = 0
        self.steps: list[dict] = []
        self.step_idx: int = -1
        self.order_text: str = ""
        self.order_num: int = 0
        self.history: list[dict] = []
        self.total_reshuffles: int = 0
        self.total_travel: float = 0.0
        self.total_reward: float = 0.0
        self.orders_done: int = 0

    # ── Agent ─────────────────────────────────────────────────────────────────

    def _predict(self, mask) -> int:
        return self.policy.act(self.env)

    # ── Order generation — inbound preferred ─────────────────────────────────

    def _random_order_type(self) -> str | None:
        occ = self.env._yard_occupancy()
        cap = self.env.grid.size
        can_in, can_out = occ < cap, occ > 0
        if can_in and can_out:
            return "inbound" if self.rng.random() < 0.7 else "outbound"
        return "inbound" if can_in else ("outbound" if can_out else None)

    # ── Step planning — runs reshuffler on a copy, never touches the real grid ─

    def _plan(self, action) -> list[dict]:
        decoded = self.env.decode_action(action)

        if decoded[0] == "inbound":
            _, bl, ba, r = decoded
            h = self.env._stack_height(bl, ba, r)
            return [{"type": "inbound", "pos": (bl, ba, r, h),
                     "text": (f"Place incoming container at  "
                              f"Block {bl+1} · Bay {ba+1} · Row {r+1} · Tier {h+1}")}]

        _, bl, ba, r, h = decoded
        # Run reshuffler on a grid copy to capture the move sequence
        saved = self.env.grid
        self.env.grid = saved.copy()
        moves = self.env.reshuffler.relocate_above(bl, ba, r, h)
        self.env.grid = saved   # restore real grid

        steps: list[dict] = []
        for mv in moves:
            _, sb, sba, sr, sh, db, dba, dr, dh = mv
            dwell = int(saved[sb, sba, sr, sh])
            steps.append({
                "type": "reshuffle", "dwell": dwell,
                "from": (sb, sba, sr, sh),
                "to":   (db, dba, dr, dh),
                "text": (f"Move container (dwell {dwell})  "
                         f"B{sb+1}·Bay{sba+1}·R{sr+1}·T{sh+1}"
                         f"  →  "
                         f"B{db+1}·Bay{dba+1}·R{dr+1}·T{dh+1}"),
            })

        ret_dwell = int(saved[bl, ba, r, h])
        reshuffle_dists = [mv[0] for mv in moves]
        steps.append({
            "type": "retrieve", "pos": (bl, ba, r, h), "dwell": ret_dwell,
            "reshuffle_dists": reshuffle_dists,
            "text": (f"Take container (dwell {ret_dwell} ticks) from  "
                     f"Block {bl+1} · Bay {ba+1} · Row {r+1} · Tier {h+1}"),
        })
        return steps

    # ── Step application ──────────────────────────────────────────────────────

    def _apply(self, step: dict):
        t = step["type"]
        if t == "inbound":
            bl, ba, r, h = step["pos"]
            self.env.grid[bl, ba, r, h] = 0
            reward, _ = self.env.reward_calc.compute_inbound_step(bl, ba, r)
            self.total_reward += reward
        elif t == "reshuffle":
            sb, sba, sr, sh = step["from"]
            db, dba, dr, dh = step["to"]
            self.env.grid[sb, sba, sr, sh] = -1
            self.env.grid[db, dba, dr, dh] = step["dwell"]
            self.total_reshuffles += 1
        elif t == "retrieve":
            bl, ba, r, h = step["pos"]
            dist = self.env.crane_dist(self.env.crane_pos, (bl, ba, r))
            self.env.grid[bl, ba, r, h] = -1
            self.env.crane_pos = np.array([bl, ba, r], dtype=np.int32)
            self.total_travel += dist
            reward, _ = self.env.reward_calc.compute_outbound_step(
                step["dwell"], dist, step["reshuffle_dists"], (bl, ba, r))
            self.total_reward += reward
        self.history.append(step)

    def _flush(self):
        """Apply all remaining unapplied steps (used when Complete skips ahead)."""
        while self.step_idx < len(self.steps):
            self._apply(self.steps[self.step_idx])
            self.step_idx += 1

    # ── Public API ────────────────────────────────────────────────────────────

    def advance_tick(self):
        """Age all containers by one tick (called manually via the sidebar button)."""
        self.env.grid[self.env.grid >= 0] += 1
        self.env.tick += 1
        self.total_reward += self.env.reward_calc.tick_penalty()

    def press_complete(self) -> bool:
        """Finish the current order (apply remaining steps) and start the next."""
        if self.step_idx >= 0:
            self._flush()
            self.orders_done += 1

        order_type = self._random_order_type()
        if order_type is None:
            return False

        self.env.mode = 0 if order_type == "inbound" else 1
        self.env.n_in_remaining  = 1 if order_type == "inbound" else 0
        self.env.n_out_remaining = 0 if order_type == "inbound" else 1
        mask = self.env.action_masks()
        if not mask.any():
            return False

        action = self._predict(mask)
        self.order_text = order_type
        self.order_num += 1
        self.steps = self._plan(action)
        self.step_idx = 0          # show first step (not yet applied)
        return True

    def press_next(self):
        """Apply the current pending step and advance to the next."""
        if self.has_next:      # no-op before the first order (step_idx == -1) or when done
            self._apply(self.steps[self.step_idx])
            self.step_idx += 1

    # ── View properties ───────────────────────────────────────────────────────

    @property
    def current_step(self) -> dict | None:
        """Step currently being shown (pending, not yet applied), or None."""
        if self.step_idx < 0 or self.step_idx >= len(self.steps):
            return None
        return self.steps[self.step_idx]

    @property
    def order_done(self) -> bool:
        return bool(self.steps) and self.step_idx >= len(self.steps)

    @property
    def has_next(self) -> bool:
        return 0 <= self.step_idx < len(self.steps)
