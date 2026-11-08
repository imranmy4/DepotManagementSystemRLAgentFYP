"""Config: INI file <-> dataclasses <-> CLI overrides, plus the named experiments.

The dataclasses (DepotConfig, RewardConfig, TrainConfig) are the single source of
truth for field names, types, and defaults. This module just moves values between
them, a config.ini, and argparse — it hardcodes no field names itself.

Precedence (lowest to highest):
    dataclass defaults  <  config.ini  <  (experiment delta)  <  CLI override

Typical use:
    parser = read_ini()                                      # configs/config.ini
    depot  = load_dataclass(DepotConfig, parser, "depot")   # ini, or defaults
    add_cli_args(ap, DepotConfig)                            # --n_blocks, ...
    depot  = apply_cli(depot, args)                          # CLI overrides
"""
import os
import configparser
from dataclasses import dataclass, fields, replace
from typing import get_type_hints

from depot import paths
from depot.env import DepotConfig
from depot.reward import RewardConfig


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _cast(value, typ):
    """Cast a string from configparser/argparse to the dataclass field type."""
    if isinstance(value, typ):
        return value
    if typ is bool:
        return _str2bool(value)
    return typ(value)


def read_ini(path=None):
    parser = configparser.ConfigParser()
    parser.read(path or paths.CONFIG_INI)
    return parser


def load_dataclass(dc_type, parser, section):
    """Build a dc_type instance from [section]; absent keys fall back to defaults.

    Raises if the section contains a key that isn't a field of the dataclass —
    that's almost always a typo and silently ignoring it would be worse.
    """
    hints = get_type_hints(dc_type)
    kwargs = {}
    if parser.has_section(section):
        for key, val in parser.items(section):
            if key not in hints:
                raise KeyError(f"[{section}] has unknown key '{key}' (not a {dc_type.__name__} field)")
            kwargs[key] = _cast(val, hints[key])
    return dc_type(**kwargs)


def add_cli_args(parser, dc_type):
    """Add an optional --<field> argument for every field of dc_type.

    Defaults are None so apply_cli can tell "user passed it" from "left unset".
    """
    hints = get_type_hints(dc_type)
    for name, typ in hints.items():
        argtype = _str2bool if typ is bool else typ
        parser.add_argument(f"--{name}", type=argtype, default=None,
                            help=f"override {name} ({typ.__name__})")


def apply_cli(instance, args):
    """Return `instance` with any provided --field CLI overrides applied."""
    overrides = {f.name: getattr(args, f.name) for f in fields(instance)
                 if getattr(args, f.name, None) is not None}
    return replace(instance, **overrides) if overrides else instance


def dump_configs(path, sections):
    """Write {section_name: dataclass_instance} to an ini file (config snapshot)."""
    parser = configparser.ConfigParser()
    for section, inst in sections.items():
        parser[section] = {f.name: str(getattr(inst, f.name)) for f in fields(inst)}
    with open(path, "w") as fh:
        parser.write(fh)


@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
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
    # Early-stop the epoch loop once the policy moves more than this KL from the old one.
    # Main guard against the high-approx_kl / high-clip_fraction thrashing PPO falls into
    # on long-episode, large-reward regimes. Set 0 to disable (-> None in train()).
    target_kl: float = 0.03
    # Hidden layer sizes for both the policy and value MLP heads, comma-separated.
    net_arch: str = "256,256"
    vf_coef: float = 0.5   # value function loss coefficient (relative to policy loss)
    # Intra-op threads for the PyTorch update. More is not faster for a 256x256 MLP:
    # measured on a 6P+4E / 16-thread laptop, 4 threads beat the default (8) by ~30%
    # and 16 was ~5x slower. 0 = leave PyTorch's default.
    torch_threads: int = 4


# Varied conditions per episode: fill 30-85%, half the episodes start from a messy
# mixed-age yard, inbound/outbound caps each shifted by up to ±3.
ROBUST_DEPOT = dict(warmstart=0.3, warmstart_max=0.85, warmstart_aged=0.5, demand_jitter=3)

# Lean reward: drop terms that overlap with others. Kept: reshuffle (scope-scaled), burial
# (capped), dwell on retrieval + per-tick mean idle, cliff, and a linear travel penalty
# (threshold_divisor=100 makes the threshold 1, so every move costs -w_travel*dist and the
# short-move compensation never applies).
LEAN_REWARD = dict(w_height=0.0, w_profile=0.0, w_center=0.0, w_max_idle=0.0,
                   w_travel_compensate=0.0, threshold_divisor=100)

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
    "abl_shape_off":       {"reward": dict(w_cliff=0.0, w_profile=0.0, w_center=0.0)},
    "abl_center_off":      {"reward": dict(w_center=0.0)},

    # Robust training on varied conditions, with the full reward and with the lean reward.
    "robust":      {"depot": ROBUST_DEPOT},
    "robust_lean": {"depot": ROBUST_DEPOT, "reward": LEAN_REWARD},
    "abl_burial_uncapped": {"reward": dict(burial_cap=0.0)},
}


def resolve_config(experiment, args=None, ini_path=None):
    """Build (DepotConfig, RewardConfig, TrainConfig) for an experiment.

    Layers: config.ini base -> experiment delta -> CLI overrides (if args given).
    """
    parser = read_ini(ini_path)
    depot = load_dataclass(DepotConfig, parser, "depot")
    reward = load_dataclass(RewardConfig, parser, "reward")
    train_cfg = load_dataclass(TrainConfig, parser, "train")

    delta = EXPERIMENTS[experiment]
    depot = replace(depot, **delta.get("depot", {}))
    reward = replace(reward, **delta.get("reward", {}))
    train_cfg = replace(train_cfg, **delta.get("train", {}))

    if args is not None:
        depot = apply_cli(depot, args)
        reward = apply_cli(reward, args)
        train_cfg = apply_cli(train_cfg, args)

    return depot, reward, train_cfg


def load_eval_config(experiment, run_dir):
    """(DepotConfig, RewardConfig) a run was trained with: its config_used.ini snapshot if
    present, else resolved from configs/config.ini + the experiment delta."""
    snap = os.path.join(run_dir, "config_used.ini")
    if os.path.exists(snap):
        p = read_ini(snap)
        return load_dataclass(DepotConfig, p, "depot"), load_dataclass(RewardConfig, p, "reward")
    depot, reward, _ = resolve_config(experiment)
    return depot, reward
