"""Heuristic baseline policies for the depot env.

Each policy maps the *unwrapped* env state to a single valid action index,
respecting the current action mask. They are greedy heuristics, each optimising
one of the three competing objectives the agent must trade off:

  - random   : uninformed floor — uniform over valid actions
  - fifo     : minimise idle time — retrieve the oldest container
  - nearest  : minimise crane travel — retrieve the container nearest the crane

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
        cb, cba, cr = env.crane_pos
        dists = [abs(bl - cb) + abs(ba - cba) + abs(r - cr)
                 for _, bl, ba, r, h in (env.decode_action(a) for a in valid)]
        return _pick_best(valid, dists, rng, maximize=False)
    return _shortest_stack_inbound(env, valid, rng)
