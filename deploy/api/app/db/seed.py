"""Seed the yard layout (blocks, slots, crane row) from the deployed model's DepotConfig.

Run from deploy/api:  python -m app.db.seed
"""
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from depot import paths
from depot.config import load_run_config
from depot.env import DepotConfig, DepotEnv

from app.db.models import Slot, YardBlock, YardState
from app.db.session import SessionLocal


class LayoutMismatch(RuntimeError):
    """The yard layout in the database differs from the model's."""


def model_depot_config() -> DepotConfig:
    """DepotConfig of the deployed model (deploy/model/config_used.ini)."""
    depot_cfg, _ = load_run_config(paths.DEPLOY_MODEL_DIR)
    return depot_cfg


def model_source() -> str:
    """Which run the deployed model was promoted from, e.g. "run_20261101-041443/best_model"."""
    return (paths.DEPLOY_MODEL_DIR / "SOURCE").read_text().strip()


def _expected_blocks(cfg: DepotConfig) -> list[tuple[int, int, int, int, int, int]]:
    """(block_id, bays, rows, tiers, grid_row, grid_col) per block, positions as the env sees them."""
    env = DepotEnv(cfg)
    return [(b, cfg.n_bays, cfg.n_rows, cfg.n_tiers, *map(int, env.block_coords(b)))
            for b in range(cfg.n_blocks)]


def _expected_slots(cfg: DepotConfig) -> set[tuple[int, int, int, int]]:
    return {(b, ba, r, t) for b in range(cfg.n_blocks) for ba in range(cfg.n_bays)
            for r in range(cfg.n_rows) for t in range(cfg.n_tiers)}


def layout_diff(session: Session, cfg: DepotConfig) -> str | None:
    """How the database layout differs from `cfg`, or None if it matches."""
    blocks = [tuple(row) for row in session.execute(
        select(YardBlock.block_id, YardBlock.num_bays, YardBlock.num_rows, YardBlock.num_tiers,
               YardBlock.grid_row, YardBlock.grid_col)
        .order_by(YardBlock.block_id))]
    if blocks != _expected_blocks(cfg):
        return (f"blocks (id, bays, rows, tiers, grid_row, grid_col) {blocks} "
                f"!= model {_expected_blocks(cfg)}")

    slots = {tuple(row) for row in session.execute(
        select(Slot.block_id, Slot.bay, Slot.row, Slot.tier))}
    expected = _expected_slots(cfg)
    if slots != expected:
        return (f"slots differ: {len(expected - slots)} missing, "
                f"{len(slots - expected)} not in the model")

    if session.get(YardState, 1) is None:
        return "yard_state row missing"
    return None


def seed_layout(session: Session, cfg: DepotConfig) -> bool:
    """Create the layout on an empty yard (True), or confirm an existing one matches (False).

    Raises LayoutMismatch if the existing layout differs.
    """
    if session.scalar(select(func.count()).select_from(YardBlock)) == 0:
        session.execute(insert(YardBlock), [
            {"block_id": b, "num_bays": ba, "num_rows": r, "num_tiers": t,
             "grid_row": gr, "grid_col": gc}
            for b, ba, r, t, gr, gc in _expected_blocks(cfg)])
        session.execute(insert(Slot), [
            {"block_id": b, "bay": ba, "row": r, "tier": t}
            for b, ba, r, t in sorted(_expected_slots(cfg))])
        session.add(YardState(id=1, crane_block=0, crane_bay=0, crane_row=0))
        session.flush()
        return True

    diff = layout_diff(session, cfg)
    if diff:
        raise LayoutMismatch(diff)
    return False


def main():
    cfg = model_depot_config()
    try:
        with SessionLocal.begin() as session:
            seeded = seed_layout(session, cfg)
    except LayoutMismatch as e:
        raise SystemExit(f"Layout mismatch, nothing changed: {e}")
    n_slots = cfg.n_blocks * cfg.n_bays * cfg.n_rows * cfg.n_tiers
    print(f"{'Seeded' if seeded else 'Already seeded'}: {cfg.n_blocks} blocks "
          f"({cfg.block_rows}×{cfg.n_blocks // cfg.block_rows} grid) × "
          f"{cfg.n_bays} bays × {cfg.n_rows} rows × {cfg.n_tiers} tiers = {n_slots} slots "
          f"(model {model_source()})")


if __name__ == "__main__":
    main()
