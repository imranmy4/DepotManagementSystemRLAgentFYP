"""Load a trained run (model, VecNormalize stats, config) and pick actions with it.

    policy = Policy.load("full")               # CURRENT run of an experiment
    policy = Policy.load("full", run="run_…")  # a specific run (id or path)
    env = DepotEnv(policy.depot_config, reward_config=policy.reward_config)
    action = policy.act(env)                   # masked, deterministic by default
"""
import os
from dataclasses import dataclass

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from depot import runs
from depot.config import load_eval_config
from depot.env import DepotEnv, DepotConfig
from depot.reward import RewardConfig


@dataclass
class Policy:
    model: MaskablePPO
    vecnorm: VecNormalize
    depot_config: DepotConfig
    reward_config: RewardConfig
    run_dir: str
    model_name: str          # e.g. "best_model" / "depot_ppo_final"

    @classmethod
    def load(cls, experiment, run=None, depot_config=None):
        """Load the default model of a run (best_model, else final — see runs.resolve_model).

        `depot_config` overrides the run's saved depot config (e.g. stress scenarios). Only
        change demand / initial-state fields: changing yard dimensions breaks the model.
        Raises FileNotFoundError if the run has no loadable model.
        """
        run_dir = runs.resolve_run_dir(experiment, run)
        saved_depot, reward_config = load_eval_config(experiment, run_dir)
        depot_config = depot_config or saved_depot
        model_path, vecnorm_path = runs.resolve_model(run_dir)
        if model_path is None:
            raise FileNotFoundError(f"no loadable model (+ normalizer) in {run_dir}")

        vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: DepotEnv(depot_config)]))
        vecnorm.training = False
        model = MaskablePPO.load(model_path)
        return cls(model, vecnorm, depot_config, reward_config, run_dir,
                   os.path.basename(model_path))

    def act(self, env, obs=None, deterministic=True) -> int:
        """Action for the env's current state. `obs` defaults to the env's raw observation."""
        if obs is None:
            obs = env._get_obs()
        nobs = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(nobs, action_masks=env.action_masks(),
                                       deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0])
