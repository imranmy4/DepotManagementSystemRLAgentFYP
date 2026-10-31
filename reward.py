import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Per-step engagement
    w_action_bonus:      float = 0.1

    # Inbound
    w_burial:            float = 0.5
    w_height:            float = 0.2
    # Cap on the oldest dwell used by the burial penalty (0 = uncapped). Burying a
    # container costs ~one future reshuffle regardless of its age; uncapped, burial grows
    # with age and swamps the shape terms in a mature yard (fresh towers always win).
    burial_cap:          float = 0.0

    # Outbound
    w_dwell:             float = 1.0
    w_travel:            float = 0.3
    w_travel_compensate: float = 0.1
    threshold_divisor:   int   = 2
    w_reshuffle:         float = 1.0
    # Reshuffle scope multipliers (cost grows the farther a container is moved).
    # Set all three equal to recover a pure count-based penalty (ablation).
    reshuffle_mult_same:     float = 1.0  # destination in the source block
    reshuffle_mult_adjacent: float = 2.0  # destination one block away
    reshuffle_mult_far:      float = 4.0  # destination >= 2 blocks away (yard-wide fallback)

    # Stack shape / stability (both 0 = off, recovers the shape-blind reward).
    # Cliff: penalise height gaps to in-block neighbours beyond cliff_tol tiers —
    # towers on inbound, pits on outbound. Measured on the stack the agent touched.
    w_cliff:             float = 0.0
    cliff_tol:           int   = 1
    # Profile: centre-heavy pyramid. A stack's height cap drops profile_step tiers per
    # ring towards the block edge; inbound above the cap is penalised per excess tier.
    w_profile:           float = 0.0
    profile_step:        int   = 1
    # Centre tie-breaker: small inbound bonus scaled by ring depth (0 at the block edge,
    # w_center in the innermost ring). Kept well below w_height / w_burial / w_cliff so it
    # only decides between otherwise-equal stacks (e.g. a flat block) — with cliff_tol=1
    # this grows blocks layer by layer with the centre one tier ahead.
    w_center:            float = 0.0

    # Per-tick (global)
    w_mean_idle:         float = 0.1
    w_max_idle:          float = 0.2

    # Per-episode
    w_episode_bonus:     float = 10.0

    # Edge case
    w_full_drop:         float = 50.0


class RewardCalculator:
    """
    Computes reward components and returns both the total and a breakdown
    (useful for logging and debugging reward shaping).
    """

    def __init__(self, config: RewardConfig, env):
        self.config = config
        self.env = env
        # Precompute max possible Manhattan distance for travel threshold
        b_w = getattr(env.config, "block_weight", 1.0)
        ba_w = getattr(env.config, "bay_weight", 1.0)
        r_w = getattr(env.config, "row_weight", 1.0)
        # Farthest block pair on the block grid: opposite corners.
        max_block_dist = (env.config.block_rows - 1) + (env.block_cols - 1)
        self.max_dist = (
            max_block_dist * b_w +
            (env.Ba - 1) * ba_w +
            (env.R - 1) * r_w
        )
        self.travel_threshold = max(1.0, self.max_dist / config.threshold_divisor)
        self.profile_caps = profile_caps(env.Ba, env.R, env.H, config.profile_step)
        ring, inner = block_rings(env.Ba, env.R)
        self.center_depth = ring / inner if inner > 0 else np.zeros_like(ring, dtype=float)

    # ─── Per-step components ─────────────────────────────────────────

    def action_bonus(self) -> float:
        return self.config.w_action_bonus

    # ─── Inbound components ──────────────────────────────────────────

    def burial_penalty(self, bl: int, ba: int, r: int) -> float:
        """Penalty proportional to oldest container already in target stack."""
        stack = self.env.grid[bl, ba, r, :]
        occupied = stack[stack >= 0]
        if len(occupied) == 0:
            return 0.0
        oldest = float(occupied.max())
        if self.config.burial_cap > 0:
            oldest = min(oldest, self.config.burial_cap)
        return -self.config.w_burial * (oldest + 1) # add 1 so penalty is nonzero even if oldest is 0 (new container)

    def height_penalty(self, bl: int, ba: int, r: int) -> float:
        """Penalty for placing onto an already-tall stack."""
        h = int((self.env.grid[bl, ba, r, :] >= 0).sum())
        return -self.config.w_height * h

    # ─── Shape components (evaluated on the post-action grid) ────────

    def cliff_penalty(self, bl: int, ba: int, r: int, outbound: bool) -> float:
        """Penalty for height gaps > cliff_tol between stack (bl, ba, r) and its in-block
        neighbours. Inbound: the stack standing above its neighbours (tower).
        Outbound: the stack sunk below its neighbours (pit)."""
        if self.config.w_cliff == 0:
            return 0.0
        h = (self.env.grid[bl] >= 0).sum(axis=2)            # (Ba, R) heights in this block
        hs = int(h[ba, r])
        excess = 0
        for nba, nr in ((ba - 1, r), (ba + 1, r), (ba, r - 1), (ba, r + 1)):
            if 0 <= nba < self.env.Ba and 0 <= nr < self.env.R:
                gap = hs - int(h[nba, nr]) if not outbound else int(h[nba, nr]) - hs
                excess += max(0, gap - self.config.cliff_tol)
        return -self.config.w_cliff * excess

    def profile_penalty(self, bl: int, ba: int, r: int) -> float:
        """Penalty per tier the stack exceeds its pyramid cap (inbound)."""
        if self.config.w_profile == 0:
            return 0.0
        h = int((self.env.grid[bl, ba, r, :] >= 0).sum())
        return -self.config.w_profile * max(0, h - int(self.profile_caps[ba, r]))

    def center_bonus(self, ba: int, r: int) -> float:
        """Small inbound bonus for centre stacks (tie-breaker), 0 on the block edge."""
        return self.config.w_center * float(self.center_depth[ba, r])

    # ─── Outbound components ─────────────────────────────────────────

    def dwell_reward(self, dwell: float) -> float:
        """Reward proportional to dwell time of the retrieved container.

        Takes the dwell value directly — the caller must read it BEFORE the
        container is removed from the grid (the slot is -1 afterwards).
        """
        return self.config.w_dwell * max(0.0, float(dwell))

    def travel_reward(self, dist: float) -> float:
        """
        Compensation if short, penalty if long.
        threshold = max_dist / threshold_divisor
        """
        if dist < self.travel_threshold:
            return self.config.w_travel_compensate * (self.max_dist - dist)
        else:
            return -self.config.w_travel * dist

    def reshuffle_penalty(self, block_dists) -> float:
        """Penalty scaled by how far each relocated container was moved.

        `block_dists` is the per-relocation |dest_block - src_block| reported by
        the reshuffler. Same block is cheapest, adjacent costs more, farther most.
        """
        total = 0.0
        for d in block_dists:
            if d == 0:
                total += self.config.reshuffle_mult_same
            elif d == 1:
                total += self.config.reshuffle_mult_adjacent
            else:
                total += self.config.reshuffle_mult_far
        return -self.config.w_reshuffle * total

    # ─── Per-tick components ─────────────────────────────────────────

    def tick_penalty(self) -> float:
        """Mean + max idle penalty applied at end of each tick."""
        occupied = self.env.grid[self.env.grid >= 0]
        if len(occupied) == 0:
            return 0.0
        mean_d = float(occupied.mean())
        max_d  = float(occupied.max())
        return -(
            self.config.w_mean_idle * mean_d +
            self.config.w_max_idle  * max_d
        )

    # ─── Per-episode ─────────────────────────────────────────────────

    def episode_bonus(self) -> float:
        return self.config.w_episode_bonus

    # ─── Edge case ───────────────────────────────────────────────────

    def full_drop_penalty(self) -> float:
        return -self.config.w_full_drop

    # ─── Composite: step-level inbound/outbound ──────────────────────

    def compute_inbound_step(self, bl, ba, r) -> tuple[float, dict]:
        breakdown = {
            "action_bonus": self.action_bonus(),
            "burial":       self.burial_penalty(bl, ba, r),
            "height":       self.height_penalty(bl, ba, r),
            "cliff":        self.cliff_penalty(bl, ba, r, outbound=False),
            "profile":      self.profile_penalty(bl, ba, r),
            "center":       self.center_bonus(ba, r),
        }
        return sum(breakdown.values()), breakdown

    def compute_outbound_step(self, retrieved_dwell, dist, reshuffle_dists, stack) -> tuple[float, dict]:
        """`stack` = (bl, ba, r) retrieved from; the cliff (pit) term is measured there."""
        breakdown = {
            "action_bonus": self.action_bonus(),
            "dwell":        self.dwell_reward(retrieved_dwell),
            "travel":       self.travel_reward(dist),
            "reshuffle":    self.reshuffle_penalty(reshuffle_dists),
            "cliff":        self.cliff_penalty(*stack, outbound=True),
        }
        return sum(breakdown.values()), breakdown

def profile_caps(Ba: int, R: int, H: int, step: int) -> np.ndarray:
    """(Ba, R) pyramid height caps for one block: H in the innermost ring, dropping
    `step` tiers per ring towards the edge (never below 1). Ring 0 = outer edge.
    e.g. 4x4, H=6, step=1 -> outer 12 stacks cap 5, inner 2x2 cap 6."""
    ring, inner = block_rings(Ba, R)
    return np.maximum(1, H - step * (inner - ring))


def block_rings(Ba: int, R: int):
    """(Ba, R) ring index of each stack in a block (0 = outer edge) and the innermost
    ring index. e.g. 4x4 -> outer 12 stacks ring 0, inner 2x2 ring 1."""
    ba = np.arange(Ba).reshape(Ba, 1)
    r = np.arange(R).reshape(1, R)
    ring = np.minimum(np.minimum(ba, Ba - 1 - ba), np.minimum(r, R - 1 - r))
    return ring, (min(Ba, R) - 1) // 2
