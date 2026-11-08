"""Tests for depot.ops.LiveOrderRun step planning and application."""
import numpy as np

from depot.env import DepotConfig
from depot.ops import LiveOrderRun

CFG = DepotConfig(n_blocks=2, block_rows=1, n_bays=2, n_rows=2, n_tiers=3, n_ticks=10,
                  max_in=2, max_out=2)


class _FirstValid:
    """Policy stub returning the first valid action."""
    def act(self, env, obs=None, deterministic=True):
        return int(np.flatnonzero(env.action_masks())[0])


def _run(warmstart=0.5, seed=0):
    return LiveOrderRun(CFG, None, _FirstValid(), warmstart=warmstart, seed=seed)


def test_steps_end_with_place_or_retrieve():
    run = _run()
    for _ in range(30):
        assert run.press_complete()
        kinds = [s["type"] for s in run.steps]
        assert kinds[-1] in ("inbound", "retrieve")
        assert all(k == "reshuffle" for k in kinds[:-1])


def test_planning_does_not_touch_grid():
    run = _run()
    before = run.env.grid.copy()
    run.press_complete()                  # plans the first order, applies nothing yet
    assert np.array_equal(run.env.grid, before)


def test_container_count_conserved():
    run = _run()
    for _ in range(40):
        n_before = int((run.env.grid >= 0).sum())
        assert run.press_complete()
        order = run.order_text
        while run.has_next:
            run.press_next()
        n_after = int((run.env.grid >= 0).sum())
        assert n_after == n_before + (1 if order == "inbound" else -1)
        assert run.total_reshuffles >= 0


def test_reshuffled_containers_keep_dwell_and_no_floating():
    run = _run(warmstart=0.7, seed=3)
    H = CFG.n_tiers
    for _ in range(40):
        assert run.press_complete()
        while run.has_next:
            step = run.current_step
            if step["type"] == "reshuffle":
                sb, sba, sr, sh = step["from"]
                assert run.env.grid[sb, sba, sr, sh] == step["dwell"]
            run.press_next()
            if step["type"] == "reshuffle":
                db, dba, dr, dh = step["to"]
                assert run.env.grid[db, dba, dr, dh] == step["dwell"]
        occ = run.env.grid >= 0
        # no container above an empty slot
        for h in range(1, H):
            assert not (occ[..., h] & ~occ[..., h - 1]).any()
