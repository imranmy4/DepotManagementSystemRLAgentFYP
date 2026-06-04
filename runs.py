"""Per-run model folder layout + resolution — shared by train.py, evaluate.py, and the
future dashboard, so they all agree on where a run's artifacts live and which is "current".

Layout:
    models/<exp>/
        <run_id>/                    # run_id = "run_YYYYmmdd-HHMMSS"
            config_used.ini          # resolved config snapshot (reproduce the env/reward)
            best_model.zip           # best eval-reward checkpoint (MaskableEvalCallback)
            best_vec_normalize.pkl   #   its matching normalizer (saved on each new best)
            depot_ppo_<N>_steps.zip            # periodic checkpoints (CheckpointCallback)
            depot_ppo_vecnormalize_<N>_steps.pkl  #   their matching normalizers
            depot_ppo_final.zip      # policy as-left when learn() finished
            vec_normalize.pkl        #   final normalizer
            evaluations.npz          # eval-reward history
        CURRENT                      # text file naming the default run_id

Why a CURRENT pointer (not a flat current_model.zip): a model is only usable together with
its normalizer + config, which live in the run dir — pointing at the dir keeps the bundle
together and avoids the "model without normalizer" footgun.
"""
import os
import glob
import time


def new_run_id():
    return time.strftime("run_%Y%m%d-%H%M%S")


def exp_dir(experiment):
    return os.path.join("models", experiment)


def write_current(experiment, run_id):
    """Point CURRENT at run_id (the default run for eval/dashboard)."""
    with open(os.path.join(exp_dir(experiment), "CURRENT"), "w") as fh:
        fh.write(run_id)


def latest_run_dir(experiment):
    runs = sorted(glob.glob(os.path.join(exp_dir(experiment), "run_*")))
    return runs[-1] if runs else None


def resolve_run_dir(experiment, run=None):
    """Directory holding the run to load. Precedence:
    explicit `run` (id or path) > CURRENT pointer > newest run_* subdir >
    legacy flat models/<exp>/ (pre-per-run layout, for backward compatibility).
    """
    ed = exp_dir(experiment)
    if run:
        return run if os.path.isdir(run) else os.path.join(ed, run)
    cur = os.path.join(ed, "CURRENT")
    if os.path.exists(cur):
        rid = open(cur).read().strip()
        d = os.path.join(ed, rid)
        if os.path.isdir(d):
            return d
    latest = latest_run_dir(experiment)
    return latest if latest else ed


def resolve_model(run_dir):
    """(model_path_without_ext, vecnormalize_path) for the default model in a run.

    Prefers best_model (+ best_vec_normalize, falling back to the run's vec_normalize)
    over depot_ppo_final. Returns (None, None) if nothing loadable is present.
    """
    best = os.path.join(run_dir, "best_model")
    final = os.path.join(run_dir, "depot_ppo_final")
    run_vn = os.path.join(run_dir, "vec_normalize.pkl")
    if os.path.exists(best + ".zip"):
        best_vn = os.path.join(run_dir, "best_vec_normalize.pkl")
        vn = best_vn if os.path.exists(best_vn) else run_vn
        if os.path.exists(vn):
            return best, vn
    if os.path.exists(final + ".zip") and os.path.exists(run_vn):
        return final, run_vn
    return None, None


def latest_checkpoint(run_dir, name_prefix="depot_ppo"):
    """(model_path_without_ext, vecnormalize_path_or_None, n_steps) for the highest-step
    checkpoint in run_dir, or (None, None, 0) if there are none."""
    cks = [c for c in glob.glob(os.path.join(run_dir, f"{name_prefix}_*_steps.zip"))
           if "vecnormalize" not in os.path.basename(c)]
    if not cks:
        return None, None, 0

    def stepnum(p):
        return int(os.path.basename(p).split("_")[-2])

    latest = max(cks, key=stepnum)
    n = stepnum(latest)
    vn = os.path.join(run_dir, f"{name_prefix}_vecnormalize_{n}_steps.pkl")
    return latest[:-4], (vn if os.path.exists(vn) else None), n
