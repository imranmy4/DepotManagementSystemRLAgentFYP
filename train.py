"""Config-driven training harness for the depot agent.

Config comes from config.ini (see configloader.py); named experiments below apply
deltas on top of that base, and any field can be overridden on the CLI.

    python train.py full                       # full config from config.ini
    python train.py full --n_blocks 5 --w_reshuffle 4   # CLI overrides
    python train.py abl_reshuffle_heavy        # an ablation delta
    python train.py full --smoke               # tiny budget, just verify wiring

Precedence: dataclass defaults < config.ini < experiment delta < CLI override.
The resolved config is snapshotted to models/<name>/config_used.ini so evaluation
always uses the exact config a model was trained with.

Ablations that need env/reward/reshuffle code changes (state composition, travel
scope, MinMax priority, corridor scope, warmstart) are NOT here yet — see CLAUDE.md.
"""
import os
import argparse
from dataclasses import dataclass, replace

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from depot_env import DepotEnv, DepotConfig
from reward import RewardConfig
import configloader


@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    # MLP policy: CPU beats GPU here (tiny net, env-stepping bound). ~997 vs ~781 fps.
    device: str = "cpu"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # Small entropy bonus to keep exploring 
    ent_coef: float = 0.01
    # Hidden layer sizes for both the policy and value MLP heads, comma-separated.
    net_arch: str = "256,256"


# Direct MaskablePPO constructor kwargs (net_arch is handled separately via policy_kwargs).
PPO_KEYS = ["learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef"]

# Each experiment is a delta applied on top of the config.ini base. "full" = base.
EXPERIMENTS = {
    "full": {},

    # Sanity check: tiny depot, short budget, single env.
    "tiny": {
        "depot": dict(n_blocks=2, n_bays=2, n_rows=2, n_tiers=2, n_ticks=10, max_in=2, max_out=2),
        "train": dict(total_timesteps=100_000, n_envs=1, n_steps=512, batch_size=64),
    },

    # ---- Config-only ablations on the full config -------------------------
    "abl_reshuffle_count_only": {
        "reward": dict(reshuffle_mult_same=1.0, reshuffle_mult_adjacent=1.0, reshuffle_mult_far=1.0),
    },
    "abl_travel_div3":     {"reward": dict(threshold_divisor=3)},
    "abl_travel_div4":     {"reward": dict(threshold_divisor=4)},
    "abl_dwell_heavy":     {"reward": dict(w_dwell=2.0)},
    "abl_travel_heavy":    {"reward": dict(w_travel=0.6)},
    "abl_reshuffle_heavy": {"reward": dict(w_reshuffle=2.0)},
}


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


def resolve_config(experiment, args=None, ini_path="config.ini"):
    """Build (DepotConfig, RewardConfig, TrainConfig) for an experiment.

    Layers: config.ini base -> experiment delta -> CLI overrides (if args given).
    """
    parser = configloader.read_ini(ini_path)
    depot = configloader.load_dataclass(DepotConfig, parser, "depot")
    reward = configloader.load_dataclass(RewardConfig, parser, "reward")
    train_cfg = configloader.load_dataclass(TrainConfig, parser, "train")

    delta = EXPERIMENTS[experiment]
    depot = replace(depot, **delta.get("depot", {}))
    reward = replace(reward, **delta.get("reward", {}))
    train_cfg = replace(train_cfg, **delta.get("train", {}))

    if args is not None:
        depot = configloader.apply_cli(depot, args)
        reward = configloader.apply_cli(reward, args)
        train_cfg = configloader.apply_cli(train_cfg, args)

    return depot, reward, train_cfg


def train(name, depot, reward, train_cfg, total_timesteps=None):
    steps = total_timesteps if total_timesteps is not None else train_cfg.total_timesteps
    model_dir = f"./models/{name}/"
    os.makedirs(model_dir, exist_ok=True)

    # Snapshot the exact config used, so evaluation can reproduce it.
    configloader.dump_configs(os.path.join(model_dir, "config_used.ini"),
                              {"depot": depot, "reward": reward, "train": train_cfg})

    vec_env = make_vec_env(depot, reward, train_cfg.n_envs)
    ppo_kwargs = {k: getattr(train_cfg, k) for k in PPO_KEYS}

    # net_arch comes from config as e.g. "256,256"; same hidden sizes for actor & critic.
    layers = [int(x) for x in str(train_cfg.net_arch).split(",") if x.strip()]
    policy_kwargs = dict(net_arch=dict(pi=layers, vf=layers))

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=train_cfg.device,
        tensorboard_log=f"./logs/{name}/",
        **ppo_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // train_cfg.n_envs, 1),
        save_path=model_dir,
        name_prefix="depot_ppo",
    )

    # Periodically evaluate on a held-out env and keep the best checkpoint
    # (best_model.zip). The final model is not necessarily the best, since reward is
    # non-monotonic. The eval env is VecNormalize-wrapped so MaskableEvalCallback syncs
    # the obs-normalisation stats from the training env before each eval; norm_reward is
    # off there so the reported eval reward is the raw env reward.
    eval_env = make_vec_env(depot, reward, 1)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(25_000 // train_cfg.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=steps,
                callback=[checkpoint_cb, eval_cb], progress_bar=True)

    model.save(os.path.join(model_dir, "depot_ppo_final"))
    vec_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Training complete for '{name}' ({steps} steps) ✓")


def main():
    ap = argparse.ArgumentParser(description="Train the depot agent on a named experiment.")
    ap.add_argument("experiment", nargs="?", default="tiny", choices=list(EXPERIMENTS),
                    help="experiment to run (default: tiny)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny step budget just to verify the harness runs end-to-end")
    # Auto-generated --field overrides for every config field.
    configloader.add_cli_args(ap, DepotConfig)
    configloader.add_cli_args(ap, RewardConfig)
    configloader.add_cli_args(ap, TrainConfig)
    args = ap.parse_args()

    depot, reward, train_cfg = resolve_config(args.experiment, args)
    steps = train_cfg.n_steps * train_cfg.n_envs * 2 if args.smoke else None
    if args.smoke:
        print(f"[smoke] running '{args.experiment}' for {steps} steps to verify wiring")
    train(args.experiment, depot, reward, train_cfg, total_timesteps=steps)


if __name__ == "__main__":
    main()
