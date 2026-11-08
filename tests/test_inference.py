"""depot.inference.Policy loads a run bundle (model + normalizer + config) and acts validly.

Builds a throwaway untrained run in a temp dir, so it doesn't depend on models/.
"""
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from depot.config import dump_configs
from depot.env import DepotEnv, DepotConfig
from depot.inference import Policy
from depot.reward import RewardConfig

TINY = DepotConfig(n_blocks=2, block_rows=1, n_bays=2, n_rows=2, n_tiers=2, n_ticks=5,
                   max_in=2, max_out=2)


def _make_run(run_dir):
    venv = VecNormalize(DummyVecEnv([lambda: ActionMasker(DepotEnv(TINY), lambda e: e.action_masks())]))
    model = MaskablePPO("MlpPolicy", venv, n_steps=16, batch_size=16)
    model.save(os.path.join(run_dir, "depot_ppo_final"))
    venv.save(os.path.join(run_dir, "vec_normalize.pkl"))
    dump_configs(os.path.join(run_dir, "config_used.ini"),
                 {"depot": TINY, "reward": RewardConfig()})


def test_policy_loads_bundle_and_config(tmp_path):
    _make_run(str(tmp_path))
    policy = Policy.load("tiny", run=str(tmp_path))
    assert policy.depot_config == TINY            # config comes from the run snapshot
    assert policy.model_name == "depot_ppo_final"
    assert policy.vecnorm.training is False


def test_policy_actions_respect_mask(tmp_path):
    _make_run(str(tmp_path))
    policy = Policy.load("tiny", run=str(tmp_path))
    env = DepotEnv(policy.depot_config, reward_config=policy.reward_config)
    env.reset(seed=0)
    for _ in range(40):
        mask = env.action_masks()
        a = policy.act(env, deterministic=False)
        assert mask[a], "policy picked a masked action"
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()


def test_policy_missing_model_raises(tmp_path):
    dump_configs(os.path.join(tmp_path, "config_used.ini"),
                 {"depot": TINY, "reward": RewardConfig()})
    try:
        Policy.load("tiny", run=str(tmp_path))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a run with no model")
