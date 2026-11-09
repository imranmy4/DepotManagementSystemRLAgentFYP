"""Config: INI file <-> dataclasses <-> CLI overrides.

The dataclasses (DepotConfig, RewardConfig, TrainConfig) are the single source of
truth for field names, types, and defaults. This module just moves values between
them, a config.ini, and argparse — it hardcodes no field names itself.

Precedence (lowest to highest):
    dataclass defaults  <  config.ini  <  CLI override

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


def resolve_config(args=None, ini_path=None):
    """Build (DepotConfig, RewardConfig, TrainConfig) from config.ini plus CLI overrides."""
    parser = read_ini(ini_path)
    depot = load_dataclass(DepotConfig, parser, "depot")
    reward = load_dataclass(RewardConfig, parser, "reward")
    train_cfg = load_dataclass(TrainConfig, parser, "train")

    if args is not None:
        depot = apply_cli(depot, args)
        reward = apply_cli(reward, args)
        train_cfg = apply_cli(train_cfg, args)

    return depot, reward, train_cfg


def load_run_config(run_dir):
    """(DepotConfig, RewardConfig) a run was trained with, from its config_used.ini snapshot."""
    snap = os.path.join(run_dir, "config_used.ini")
    if not os.path.exists(snap):
        raise FileNotFoundError(f"no config_used.ini in {run_dir}")
    p = read_ini(snap)
    return load_dataclass(DepotConfig, p, "depot"), load_dataclass(RewardConfig, p, "reward")
