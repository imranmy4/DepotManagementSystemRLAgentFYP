"""Tests for the per-slot scoring policy: observation parsing, action order, masking."""
import numpy as np
import torch as th

from depot.env import DepotConfig, DepotEnv
from depot.policy import SlotNetwork

CFG = DepotConfig(warmstart=0.5, warmstart_age_mean=20)


def _net_and_env():
    env = DepotEnv(CFG)
    env.reset(seed=1)
    net = SlotNetwork((CFG.n_blocks, CFG.n_bays, CFG.n_rows, CFG.n_tiers), max_dist=9)
    return net, env


def test_logits_cover_every_action():
    net, env = _net_and_env()
    logits = net.forward_actor(th.as_tensor(env._get_obs()).unsqueeze(0))
    assert logits.shape == (1, env.n_actions)


def test_parsed_features_match_action_positions():
    """Slot features follow the env's outbound action order; stack features its inbound order."""
    net, env = _net_and_env()
    obs = th.as_tensor(env._get_obs()).unsqueeze(0)
    stack, slot, _, _ = net._features(obs)
    slot_flat, stack_flat = slot.reshape(-1, slot.shape[-1]), stack.reshape(-1, stack.shape[-1])
    heights = (env.grid >= 0).sum(axis=3)
    for a in range(env.n_inbound, env.n_actions, 37):
        _, bl, ba, r, h = env.decode_action(a)
        f = slot_flat[a - env.n_inbound]
        assert f[0].item() == float(env.grid[bl, ba, r, h] >= 0)
        if env.grid[bl, ba, r, h] >= 0:
            assert np.isclose(f[1].item(), np.log1p(env.grid[bl, ba, r, h]) / 6.0, atol=1e-5)
            assert np.isclose(f[2].item() * CFG.n_tiers, heights[bl, ba, r] - 1 - h)
    for a in range(0, env.n_inbound, 7):
        _, bl, ba, r = env.decode_action(a)
        assert np.isclose(stack_flat[a][0].item() * CFG.n_tiers, heights[bl, ba, r])


def test_retrieval_score_shared_across_slots():
    """Two slots with identical local features and context get identical scores."""
    net, env = _net_and_env()
    env.grid[:] = -1
    env.grid[0, 1, 1, 0] = 10                 # crane stack
    env.grid[0, 1, 2, 0] = 10                 # two mirror slots next to it: same age, height,
    env.grid[0, 2, 1, 0] = 10                 # neighbour heights and crane distance
    env.crane_pos = np.array([0, 1, 1])
    obs = th.as_tensor(env._get_obs()).unsqueeze(0)
    logits = net.forward_actor(obs)[0]
    i1 = env.n_inbound + 0 * 96 + 1 * 24 + 2 * 6     # (0,1,2,0)
    i2 = env.n_inbound + 0 * 96 + 2 * 24 + 1 * 6     # (0,2,1,0) — same distance, same neighbours
    assert th.isclose(logits[i1], logits[i2])


def test_deadline_features_mark_overdue_slots():
    env = DepotEnv(CFG)
    env.reset(seed=1)
    env.grid[:] = -1
    env.grid[0, 0, 0, :2] = [200, 100]          # bottom overdue, top not
    net = SlotNetwork((CFG.n_blocks, CFG.n_bays, CFG.n_rows, CFG.n_tiers), max_dist=9, max_age=150)
    stack, slot, _, _ = net._features(th.as_tensor(env._get_obs()).unsqueeze(0))
    assert slot.shape[-1] == 6 and stack.shape[-1] == 12
    assert slot[0, 0, 0, 0, 0, 4] == 1.0 and slot[0, 0, 0, 0, 1, 4] == 0.0
    assert np.isclose(slot[0, 0, 0, 0, 0, 5].item(), (200 - 150) / 150)
    assert slot[0, 0, 0, 0, 2, 4] == 0.0 and slot[0, 0, 0, 0, 2, 5] == 0.0   # empty slot
    assert stack[0, 0, 0, 0, 10] == 1.0                                       # oldest is overdue
    assert net.forward_actor(th.as_tensor(env._get_obs()).unsqueeze(0)).shape == (1, env.n_actions)
