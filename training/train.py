"""Config-driven training harness for the depot agent.

Config comes from configs/config.ini; any field can be overridden on the CLI.

    python training/train.py                          # config.ini as-is
    python training/train.py --w_reshuffle 4          # CLI override
    python training/train.py --smoke                  # tiny budget, just verify wiring
    python training/train.py --resume [run_id]        # continue a run from its checkpoint

Precedence: dataclass defaults < config.ini < CLI override.
The resolved config is snapshotted to models/<run_id>/config_used.ini so evaluation
always uses the exact config a model was trained with.
"""
import os
import argparse

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from depot import config as configloader, paths, runs
from depot.config import TrainConfig, resolve_config
from depot.env import DepotEnv, DepotConfig
from depot.reward import RewardConfig


class _SaveVecNormalize(BaseCallback):
    """Persist the training VecNormalize stats to `save_path` whenever invoked.

    Used as MaskableEvalCallback's callback_on_new_best so best_model.zip always has a
    matching normalizer — otherwise the stats are only written at the very end of training,
    leaving an aborted run (or best_model) impossible to evaluate.
    """
    def __init__(self, vec_env, save_path):
        super().__init__()
        self.vec_env = vec_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.vec_env.save(self.save_path)
        return True


# Direct MaskablePPO constructor kwargs (net_arch is handled separately via policy_kwargs).
PPO_KEYS = ["learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"]


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


def _resume_run_dir(resume):
    """Locate the run dir to resume: latest run by default, or an explicit id / path."""
    run_dir = runs.latest_run_dir() if resume == "__latest__" else (
        resume if os.path.isdir(resume) else runs.run_dir(resume))
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--resume: no run directory found for '{resume}' in models/")
    return run_dir


def train(depot, reward, train_cfg, total_timesteps=None, resume=None, set_current=True):
    target_total = total_timesteps if total_timesteps is not None else train_cfg.total_timesteps

    # Each run gets its own folder; CURRENT points at the default run for eval/dashboard.
    if resume:
        run_dir = _resume_run_dir(resume)
        run_id = os.path.basename(run_dir.rstrip("/"))
    else:
        run_id = runs.new_run_id()
        run_dir = runs.run_dir(run_id)
    os.makedirs(run_dir, exist_ok=True)
    tb_log = os.path.join(paths.LOGS_DIR, run_id)   # logs share the run_id, so logs <-> models match

    if not resume:
        # Snapshot the exact config used, so evaluation can reproduce it.
        configloader.dump_configs(os.path.join(run_dir, "config_used.ini"),
                                  {"depot": depot, "reward": reward, "train": train_cfg})

    if train_cfg.torch_threads > 0:
        torch.set_num_threads(train_cfg.torch_threads)

    vec_env = make_vec_env(depot, reward, train_cfg.n_envs)
    ppo_kwargs = {k: getattr(train_cfg, k) for k in PPO_KEYS}
    # net_arch comes from config as e.g. "256,256"; same hidden sizes for actor & critic.
    layers = [int(x) for x in str(train_cfg.net_arch).split(",") if x.strip()]
    policy_kwargs = dict(net_arch=dict(pi=layers, vf=layers))

    if resume:
        ckpt, ckpt_vn, done = runs.latest_checkpoint(run_dir)
        if ckpt is None:
            raise FileNotFoundError(f"--resume: no checkpoint in {run_dir}")
        if ckpt_vn:                                   # restore obs/reward normalisation
            vec_env = VecNormalize.load(ckpt_vn, vec_env.venv)
        model = MaskablePPO.load(ckpt, env=vec_env, device=train_cfg.device,
                                 tensorboard_log=tb_log)
        steps = max(0, target_total - done)
        print(f"[resume] run '{run_id}' from {os.path.basename(ckpt)} "
              f"({done} done) -> {steps} more to reach {target_total}")
        if steps == 0:
            print("[resume] already at target; nothing to do.")
            return
    else:
        steps = target_total
        model = MaskablePPO(
            MaskableActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs,
            target_kl=(train_cfg.target_kl or None),   # 0 -> None (disabled)
            verbose=1, device=train_cfg.device, tensorboard_log=tb_log, **ppo_kwargs,
        )

    # Checkpoints + their VecNormalize stats, so any checkpoint (and an aborted run) is
    # evaluable/resumable on its own.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // train_cfg.n_envs, 1),
        save_path=run_dir,
        name_prefix="depot_ppo",
        save_vecnormalize=True,
    )

    # Periodically evaluate on a held-out env and keep the best checkpoint (best_model.zip).
    # The final model is not necessarily the best, since reward is non-monotonic. The eval
    # env is VecNormalize-wrapped so MaskableEvalCallback syncs the obs-normalisation stats
    # from the training env before each eval; norm_reward is off there so the reported eval
    # reward is raw. callback_on_new_best snapshots the matching normalizer alongside
    # best_model so it is always evaluable.
    eval_env = make_vec_env(depot, reward, 1)
    eval_env.training = False
    eval_env.norm_reward = False
    save_best_vn = _SaveVecNormalize(vec_env, os.path.join(run_dir, "best_vec_normalize.pkl"))
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(100_000 // train_cfg.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        callback_on_new_best=save_best_vn,
    )

    model.learn(total_timesteps=steps, callback=[checkpoint_cb, eval_cb],
                reset_num_timesteps=not resume, progress_bar=True)

    model.save(os.path.join(run_dir, "depot_ppo_final"))
    vec_env.save(os.path.join(run_dir, "vec_normalize.pkl"))
    if set_current:
        runs.write_current(run_id)        # this run becomes the default for eval/dashboard
    print(f"Training complete for run '{run_id}' ({steps} steps) ✓")


def main():
    ap = argparse.ArgumentParser(description="Train the depot agent.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny step budget just to verify the harness runs end-to-end "
                         "(does not move CURRENT)")
    ap.add_argument("--resume", nargs="?", const="__latest__", default=None,
                    help="resume training from a run's latest checkpoint: latest run by "
                         "default, or pass a run id / path (e.g. --resume run_20260603-194312)")
    # Auto-generated --field overrides for every config field.
    configloader.add_cli_args(ap, DepotConfig)
    configloader.add_cli_args(ap, RewardConfig)
    configloader.add_cli_args(ap, TrainConfig)
    args = ap.parse_args()

    depot, reward, train_cfg = resolve_config(args)
    steps = train_cfg.n_steps * train_cfg.n_envs * 2 if args.smoke else None
    if args.smoke:
        print(f"[smoke] running for {steps} steps to verify wiring")
    train(depot, reward, train_cfg, total_timesteps=steps, resume=args.resume,
          set_current=not args.smoke)


if __name__ == "__main__":
    main()
