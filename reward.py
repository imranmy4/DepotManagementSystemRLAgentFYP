import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Per-step engagement
    w_action_bonus:      float = 0.1

    # Inbound
    w_burial:            float = 0.5
    w_height:            float = 0.2

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
        self.max_dist = (
            (env.B - 1) * b_w +
            (env.Ba - 1) * ba_w +
            (env.R - 1) * r_w
        )
        self.travel_threshold = max(1.0, self.max_dist / config.threshold_divisor)

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
        return -self.config.w_burial * oldest

    def height_penalty(self, bl: int, ba: int, r: int) -> float:
        """Penalty for placing onto an already-tall stack."""
        h = int((self.env.grid[bl, ba, r, :] >= 0).sum())
        return -self.config.w_height * h

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
        }
        return sum(breakdown.values()), breakdown

    def compute_outbound_step(self, retrieved_dwell, dist, reshuffle_dists) -> tuple[float, dict]:
        breakdown = {
            "action_bonus": self.action_bonus(),
            "dwell":        self.dwell_reward(retrieved_dwell),
            "travel":       self.travel_reward(dist),
            "reshuffle":    self.reshuffle_penalty(reshuffle_dists),
        }
        return sum(breakdown.values()), breakdown