"""Config-driven training harness for the depot agent.

Run a single named experiment:
    python train.py full
    python train.py tiny
    python train.py abl_reshuffle_count_only

Verify the harness wiring without a real run (tiny step budget):
    python train.py full --smoke

Override the step budget:
    python train.py full --timesteps 1000000

Experiments are defined in EXPERIMENTS below. The "full" experiment is the full
DepotConfig; the "abl_*" experiments are config-only ablations on top of it
(reward-weight / threshold sweeps that need no code changes). Ablations that
require env/reward/reshuffle code changes (state composition, travel scope,
MinMax priority, corridor scope, warmstart) are NOT in here yet.
"""
import os
import argparse
from dataclasses import dataclass, field

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from depot_env import DepotEnv, DepotConfig
from reward import RewardConfig


def mask_fn(env):
    """Required by ActionMasker — returns the boolean mask."""
    return env.action_masks()


def make_env(depot_config, reward_config):
    """Factory that wraps the env with masking + monitor."""
    def _init():
        env = DepotEnv(depot_config, reward_config=reward_config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def make_vec_env(depot_config, reward_config, n_envs):
    """Vectorised, observation/reward-normalised env stack for training."""
    venv = DummyVecEnv([make_env(depot_config, reward_config) for _ in range(n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


# Default PPO hyperparameters (per-experiment overrides via Experiment.ppo)
DEFAULT_PPO = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)


@dataclass
class Experiment:
    name: str
    depot: DepotConfig
    reward: RewardConfig
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    ppo: dict = field(default_factory=dict)


TINY = DepotConfig(n_blocks=2, n_bays=2, n_rows=2, n_tiers=2, n_ticks=10, max_in=2, max_out=2)
FULL = DepotConfig()  # defaults are the full config (B=6, Ba=4, R=4, H=6, N=30)

# Multipliers all equal -> pure count-based reshuffle penalty (scope ablation).
RESHUFFLE_COUNT_ONLY = RewardConfig(
    reshuffle_mult_same=1.0, reshuffle_mult_adjacent=1.0, reshuffle_mult_far=1.0,
)

EXPERIMENTS = {
    # Sanity check (matches the original tiny run / evaluate.py default)
    "tiny": Experiment("tiny", TINY, RewardConfig(),
                       total_timesteps=100_000, n_envs=1,
                       ppo=dict(n_steps=512, batch_size=64)),

    # Full-config baseline
    "full": Experiment("full", FULL, RewardConfig()),

    # ---- Config-only ablations on the full config -------------------------
    # Reshuffle penalty: scope-scaled (baseline) vs count-only
    "abl_reshuffle_count_only": Experiment("abl_reshuffle_count_only", FULL, RESHUFFLE_COUNT_ONLY),
    # Travel threshold divisor sweep (baseline divisor = 2)
    "abl_travel_div3": Experiment("abl_travel_div3", FULL, RewardConfig(threshold_divisor=3)),
    "abl_travel_div4": Experiment("abl_travel_div4", FULL, RewardConfig(threshold_divisor=4)),
    # Reward-weight ratio sweeps
    "abl_dwell_heavy":   Experiment("abl_dwell_heavy", FULL, RewardConfig(w_dwell=2.0)),
    "abl_travel_heavy":  Experiment("abl_travel_heavy", FULL, RewardConfig(w_travel=0.6)),
    "abl_reshuffle_heavy": Experiment("abl_reshuffle_heavy", FULL, RewardConfig(w_reshuffle=2.0)),
}


def train(exp: Experiment, total_timesteps=None):
    steps = total_timesteps if total_timesteps is not None else exp.total_timesteps
    model_dir = f"./models/{exp.name}/"
    os.makedirs(model_dir, exist_ok=True)

    vec_env = make_vec_env(exp.depot, exp.reward, exp.n_envs)
    ppo_kwargs = {**DEFAULT_PPO, **exp.ppo}
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        verbose=1,
        # MLP policy: CPU is faster than GPU here (tiny net, env-stepping bound;
        # GPU transfer overhead dominates). Benchmarked ~997 vs ~781 fps.
        device="cpu",
        tensorboard_log=f"./logs/{exp.name}/",
        **ppo_kwargs,
    )

    # Checkpoint roughly every 50k env steps (save_freq counts per-env steps)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // exp.n_envs, 1),
        save_path=model_dir,
        name_prefix="depot_ppo",
    )

    model.learn(total_timesteps=steps, callback=checkpoint_cb, progress_bar=True)

    model.save(os.path.join(model_dir, "depot_ppo_final"))
    vec_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Training complete for '{exp.name}' ({steps} steps) ✓")


def main():
    ap = argparse.ArgumentParser(description="Train the depot agent on a named experiment.")
    ap.add_argument("experiment", nargs="?", default="tiny", choices=list(EXPERIMENTS),
                    help="experiment to run (default: tiny)")
    ap.add_argument("--timesteps", type=int, default=None, help="override total timesteps")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny step budget just to verify the harness runs end-to-end")
    args = ap.parse_args()

    exp = EXPERIMENTS[args.experiment]
    steps = args.timesteps
    if args.smoke:
        steps = exp.ppo.get("n_steps", DEFAULT_PPO["n_steps"]) * exp.n_envs * 2
        print(f"[smoke] running '{exp.name}' for {steps} steps to verify wiring")
    train(exp, total_timesteps=steps)


if __name__ == "__main__":
    main()
