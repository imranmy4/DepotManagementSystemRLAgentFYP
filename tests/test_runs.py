"""Tests for run resolution and promotion of a run to the deployment model folder."""
import os

import pytest

from depot import paths, runs


def _fake_run(models, run_id, files=("config_used.ini", "best_model.zip", "best_vec_normalize.pkl")):
    d = models / run_id
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text(f)
    return d


@pytest.fixture
def models(tmp_path, monkeypatch):
    m = tmp_path / "models"
    m.mkdir()
    monkeypatch.setattr(paths, "MODELS_DIR", m)
    return m


def test_resolve_prefers_current_then_latest(models):
    _fake_run(models, "run_20260101-000000")
    _fake_run(models, "run_20260202-000000")
    assert runs.resolve_run_dir().endswith("run_20260202-000000")
    runs.write_current("run_20260101-000000")
    assert runs.resolve_run_dir().endswith("run_20260101-000000")
    assert runs.resolve_run_dir("run_20260202-000000").endswith("run_20260202-000000")


def test_resolve_missing_raises(models):
    with pytest.raises(FileNotFoundError):
        runs.resolve_run_dir()
    with pytest.raises(FileNotFoundError):
        runs.resolve_run_dir("run_nope")


def test_promote_replaces_bundle(models, tmp_path):
    dest = tmp_path / "deploy_model"
    a = _fake_run(models, "run_a", files=("config_used.ini", "best_model.zip",
                                          "best_vec_normalize.pkl", "depot_ppo_1_steps.zip"))
    assert runs.promote(a, dest) == "run_a/best_model"
    assert sorted(os.listdir(dest)) == ["SOURCE", "best_model.zip", "best_vec_normalize.pkl",
                                        "config_used.ini"]

    b = _fake_run(models, "run_b", files=("config_used.ini", "depot_ppo_final.zip",
                                          "vec_normalize.pkl"))
    assert runs.promote(b, dest) == "run_b/depot_ppo_final"
    # The previous run's best_model must not linger and shadow the new final model.
    assert sorted(os.listdir(dest)) == ["SOURCE", "config_used.ini", "depot_ppo_final.zip",
                                        "vec_normalize.pkl"]


def test_promote_without_model_raises(models, tmp_path):
    a = _fake_run(models, "run_a", files=("config_used.ini",))
    with pytest.raises(FileNotFoundError):
        runs.promote(a, tmp_path / "deploy_model")
