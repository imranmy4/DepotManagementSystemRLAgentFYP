"""Heuristic baseline policies for the depot env.

Each policy maps the *unwrapped* env state to a single valid action index,
respecting the current action mask. They are greedy heuristics, each optimising
one of the three competing objectives the agent must trade off:

  - random   : uninformed floor — uniform over valid actions
  - fifo     : minimise idle time — retrieve the oldest container
  - nearest  : minimise crane travel — retrieve the container nearest the crane
  - planner  : strong rule-of-thumb baseline balancing all three (see planner_action);
               the bar the agent must clear to show RL adds value over hand-written rules

FIFO and Nearest share the SAME inbound rule (place on the shortest stack) so
they differ only in their retrieval strategy. That isolates the outbound
tradeoff, which is the part the RL agent is meant to learn.
"""
import numpy as np


def _pick_best(valid, keys, rng, maximize):
    """Pick the valid action with the best key; break ties uniformly at random."""
    keys = np.asarray(keys, dtype=float)
    target = keys.max() if maximize else keys.min()
    tied = valid[np.isclose(keys, target)]
    return int(rng.choice(tied))


def _shortest_stack_inbound(env, valid, rng):
    """Shared inbound rule: place on the shortest (lowest) eligible stack."""
    heights = [int((env.grid[bl, ba, r, :] >= 0).sum())
               for _, bl, ba, r in (env.decode_action(a) for a in valid)]
    return _pick_best(valid, heights, rng, maximize=False)


def random_action(env, rng):
    """Uniform random over currently valid (mask-feasible) actions."""
    valid = np.flatnonzero(env.action_masks())
    return int(rng.choice(valid))


def fifo_action(env, rng):
    """First-in-first-out: retrieve the oldest container; place on shortest stack."""
    valid = np.flatnonzero(env.action_masks())
    if env.mode == 1:  # outbound -> oldest (max dwell)
        dwells = [int(env.grid[bl, ba, r, h])
                  for _, bl, ba, r, h in (env.decode_action(a) for a in valid)]
        return _pick_best(valid, dwells, rng, maximize=True)
    return _shortest_stack_inbound(env, valid, rng)


def nearest_action(env, rng):
    """Greedy crane travel: retrieve the nearest container; place on shortest stack."""
    valid = np.flatnonzero(env.action_masks())
    if env.mode == 1:  # outbound -> nearest to crane (min Manhattan)
        dists = [env.crane_dist(env.crane_pos, (bl, ba, r))
                 for _, bl, ba, r, h in (env.decode_action(a) for a in valid)]
        return _pick_best(valid, dists, rng, maximize=False)
    return _shortest_stack_inbound(env, valid, rng)


# ─── Planner: a reasonable human yard-planner's rule of thumb ────────────────
PLANNER_PROFILE_STEP = 1      # pyramid: height cap drops one tier per ring towards the block edge


def block_rings(Ba: int, R: int):
    """(Ba, R) ring index of each stack in a block (0 = outer edge) and the innermost
    ring index. e.g. 4x4 -> outer 12 stacks ring 0, inner 2x2 ring 1."""
    ba = np.arange(Ba).reshape(Ba, 1)
    r = np.arange(R).reshape(1, R)
    ring = np.minimum(np.minimum(ba, Ba - 1 - ba), np.minimum(r, R - 1 - r))
    return ring, (min(Ba, R) - 1) // 2


def profile_caps(Ba: int, R: int, H: int, step: int) -> np.ndarray:
    """(Ba, R) pyramid height caps for one block: H in the innermost ring, dropping
    `step` tiers per ring towards the edge (never below 1).
    e.g. 4x4, H=6, step=1 -> outer 12 stacks cap 5, inner 2x2 cap 6."""
    ring, inner = block_rings(Ba, R)
    return np.maximum(1, H - step * (inner - ring))


def planner_action(env, rng):
    """Strong heuristic baseline — what a sensible yard planner would do.

    Outbound: containers past the release deadline (reward max_age) go out first; otherwise, and
      among overdue ones, retrieve the container with the best immediate reward at the reward's own prices,
      age bonus + travel penalty − reshuffles (each priced as a same-block move), i.e. prefer old
      containers, dig only when the age gain outweighs the reshuffles, and prefer nearby ones.
    Inbound: obey the same safety rules the agent is rewarded for — never create a cliff
      (> cliff_tol tiers above a neighbour) or exceed the pyramid cap. Among safe stacks,
      keep fresh containers together (youngest oldest-container first), then the lowest
      stack, then prefer the block centre. If no stack is safe, take the least-bad one.
    """
    valid = np.flatnonzero(env.action_masks())
    decoded = [env.decode_action(a) for a in valid]
    heights = (env.grid >= 0).sum(axis=3)                         # (B, Ba, R)

    if env.mode == 1:
        rc = env.reward_calc
        resh_cost = rc.config.w_reshuffle * rc.config.reshuffle_mult_same
        ages = np.array([int(env.grid[bl, ba, r, h]) for _, bl, ba, r, h in decoded])
        overdue = ages > rc.config.max_age
        scores = []
        for (_, bl, ba, r, h), age in zip(decoded, ages):
            n_resh = int(heights[bl, ba, r]) - 1 - h
            dist = env.crane_dist(env.crane_pos, (bl, ba, r))
            scores.append(rc.dwell_reward(age) + rc.travel_penalty(dist) - resh_cost * n_resh)
        if overdue.any():                      # past the release deadline: these go out first
            scores = np.where(overdue, scores, -np.inf)
        return _pick_best(valid, scores, rng, maximize=True)

    tol = env.reward_calc.config.cliff_tol
    caps = profile_caps(env.Ba, env.R, env.H, PLANNER_PROFILE_STEP)
    ring, inner = block_rings(env.Ba, env.R)
    center_depth = ring / inner if inner > 0 else np.zeros_like(ring, dtype=float)
    stack_max = env.grid.max(axis=3)                              # -1 for empty stacks
    keys = []
    for _, bl, ba, r in decoded:
        new_h = int(heights[bl, ba, r]) + 1
        excess = 0
        for nba, nr in ((ba - 1, r), (ba + 1, r), (ba, r - 1), (ba, r + 1)):
            if 0 <= nba < env.Ba and 0 <= nr < env.R:
                excess += max(0, new_h - int(heights[bl, nba, nr]) - tol)
        unsafe = excess + max(0, new_h - int(caps[ba, r]))
        keys.append((unsafe, int(stack_max[bl, ba, r]), new_h, -float(center_depth[ba, r])))
    best = min(keys)
    tied = valid[[k == best for k in keys]]
    return int(rng.choice(tied))
