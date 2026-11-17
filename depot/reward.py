"""Reward terms of the depot env: burial and cliff on inbound; dwell, travel, reshuffle and cliff
on outbound; mean idle per tick."""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Inbound: penalty for burying containers, scaled by the oldest one below the new container
    # (capped: burying costs about one future reshuffle regardless of age; uncapped it swamps
    # the other terms in a mature yard). An empty stack costs nothing.
    w_burial:            float = 2.0
    burial_cap:          float = 5.0

    # Outbound
    w_dwell:             float = 0.5
    w_travel:            float = 0.3    # per unit of crane distance
    w_reshuffle:         float = 3.0
    # Reshuffle scope multipliers (cost grows the farther a container is moved).
    reshuffle_mult_same:     float = 1.0  # destination in the source block
    reshuffle_mult_adjacent: float = 2.0  # destination one block away
    reshuffle_mult_far:      float = 3.0  # destination >= 2 blocks away

    # Stack shape: penalise height gaps to in-block neighbours beyond cliff_tol tiers (towers on
    # inbound, pits on outbound), as the MEAN excess over the neighbours so centre, edge and
    # corner stacks pay the same for the same steepness.
    w_cliff:             float = 15.0
    cliff_tol:           int   = 1

    # Per tick: mean dwell of the containers in the yard.
    w_mean_idle:         float = 0.2


# Fields of earlier reward versions, still present in old run snapshots; ignored when loading.
LEGACY_REWARD_FIELDS = frozenset({
    "w_action_bonus", "w_height", "w_travel_compensate", "threshold_divisor", "w_profile",
    "profile_step", "w_center", "w_max_idle", "w_episode_bonus", "w_full_drop",
})


class RewardCalculator:
    """Computes reward components; returns the total and a per-term breakdown."""

    def __init__(self, config: RewardConfig, env):
        self.config = config
        self.env = env

    # ─── Inbound ─────────────────────────────────────────────────────

    def burial_penalty(self, bl: int, ba: int, r: int) -> float:
        """Penalty for the container just placed on (bl, ba, r), scaled by the oldest container
        below it; 0 on an empty stack. Evaluated on the post-placement grid."""
        stack = self.env.grid[bl, ba, r, :]
        below = stack[stack >= 0][:-1]          # stacks fill bottom-up: the new one is on top
        if len(below) == 0:
            return 0.0
        oldest = min(float(below.max()), self.config.burial_cap)
        return -self.config.w_burial * (oldest + 1)   # +1: burying fresh containers still costs

    # ─── Shape (post-action grid) ────────────────────────────────────

    def cliff_penalty(self, bl: int, ba: int, r: int, outbound: bool) -> float:
        """Penalty for height gaps > cliff_tol between stack (bl, ba, r) and its in-block
        neighbours, averaged over the neighbours. Inbound: the stack standing above its
        neighbours (tower). Outbound: the stack sunk below its neighbours (pit)."""
        if self.config.w_cliff == 0:
            return 0.0
        h = (self.env.grid[bl] >= 0).sum(axis=2)            # (Ba, R) heights in this block
        hs = int(h[ba, r])
        excess, n_nb = 0, 0
        for nba, nr in ((ba - 1, r), (ba + 1, r), (ba, r - 1), (ba, r + 1)):
            if 0 <= nba < self.env.Ba and 0 <= nr < self.env.R:
                gap = hs - int(h[nba, nr]) if not outbound else int(h[nba, nr]) - hs
                excess += max(0, gap - self.config.cliff_tol)
                n_nb += 1
        return -self.config.w_cliff * excess / n_nb if n_nb else 0.0

    # ─── Outbound ────────────────────────────────────────────────────

    def dwell_reward(self, dwell: float) -> float:
        """Reward proportional to the retrieved container's dwell (read before removal)."""
        return self.config.w_dwell * max(0.0, float(dwell))

    def travel_penalty(self, dist: float) -> float:
        return -self.config.w_travel * dist

    def reshuffle_penalty(self, block_dists) -> float:
        """Penalty per relocated container, scaled by its block distance (same / adjacent / far)."""
        mult = (self.config.reshuffle_mult_same, self.config.reshuffle_mult_adjacent,
                self.config.reshuffle_mult_far)
        return -self.config.w_reshuffle * sum(mult[min(int(d), 2)] for d in block_dists)

    # ─── Per tick ────────────────────────────────────────────────────

    def tick_penalty(self) -> float:
        """Mean-dwell idle penalty for one tick."""
        occupied = self.env.grid[self.env.grid >= 0]
        if len(occupied) == 0:
            return 0.0
        return -self.config.w_mean_idle * float(occupied.mean())

    # ─── Per step ────────────────────────────────────────────────────

    def compute_inbound_step(self, bl, ba, r) -> tuple[float, dict]:
        breakdown = {
            "burial": self.burial_penalty(bl, ba, r),
            "cliff":  self.cliff_penalty(bl, ba, r, outbound=False),
        }
        return sum(breakdown.values()), breakdown

    def compute_outbound_step(self, retrieved_dwell, dist, reshuffle_dists, stack) -> tuple[float, dict]:
        """`stack` = (bl, ba, r) retrieved from; the cliff (pit) term is measured there."""
        breakdown = {
            "dwell":     self.dwell_reward(retrieved_dwell),
            "travel":    self.travel_penalty(dist),
            "reshuffle": self.reshuffle_penalty(reshuffle_dists),
            "cliff":     self.cliff_penalty(*stack, outbound=True),
        }
        return sum(breakdown.values()), breakdown
