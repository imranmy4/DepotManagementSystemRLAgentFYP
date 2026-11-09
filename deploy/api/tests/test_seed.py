"""Tests for seeding the yard layout and detecting a layout that differs from the model."""
import dataclasses

import pytest
from sqlalchemy import func, select

from app.db.models import Slot, YardBlock, YardState
from app.db.seed import LayoutMismatch, layout_diff, seed_layout
from conftest import TINY


def test_seed_creates_layout(session):
    assert seed_layout(session, TINY) is True
    assert session.scalar(select(func.count()).select_from(Slot)) == 2 * 2 * 2 * 3
    assert [(b.block_id, b.grid_row, b.grid_col) for b in
            session.scalars(select(YardBlock).order_by(YardBlock.block_id))] == [(0, 0, 0), (1, 0, 1)]
    assert session.get(YardState, 1).crane_block == 0
    assert layout_diff(session, TINY) is None


def test_seed_is_idempotent(session):
    seed_layout(session, TINY)
    assert seed_layout(session, TINY) is False


@pytest.mark.parametrize("change", [dict(n_tiers=4), dict(n_blocks=3, block_rows=1),
                                    dict(block_rows=2)])
def test_seed_refuses_different_layout(session, change):
    seed_layout(session, TINY)
    with pytest.raises(LayoutMismatch):
        seed_layout(session, dataclasses.replace(TINY, **change))


def test_layout_diff_detects_changed_slot(session):
    seed_layout(session, TINY)
    session.scalars(select(Slot).order_by(Slot.slot_id.desc())).first().tier = 9
    assert "slots differ" in layout_diff(session, TINY)
