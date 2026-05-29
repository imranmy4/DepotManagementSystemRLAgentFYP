"""Generic config plumbing: INI file <-> dataclasses <-> CLI overrides.

The dataclasses (DepotConfig, RewardConfig, TrainConfig) are the single source of
truth for field names, types, and defaults. This module just moves values between
them, a config.ini, and argparse — it hardcodes no field names itself.

Precedence (lowest to highest):
    dataclass defaults  <  config.ini  <  (experiment delta)  <  CLI override

Typical use (see train.py):
    parser = read_ini("config.ini")
    depot  = load_dataclass(DepotConfig, parser, "depot")   # ini, or defaults
    add_cli_args(ap, DepotConfig)                            # --n_blocks, ...
    depot  = apply_cli(depot, args)                          # CLI overrides
"""
import configparser
from dataclasses import fields, replace
from typing import get_type_hints


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _cast(value, typ):
    """Cast a string from configparser/argparse to the dataclass field type."""
    if isinstance(value, typ):
        return value
    if typ is bool:
        return _str2bool(value)
    return typ(value)


def read_ini(path="config.ini"):
    parser = configparser.ConfigParser()
    parser.read(path)
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
