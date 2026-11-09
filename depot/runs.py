"""Per-run model folder layout, run/model resolution, and promotion to deployment.

Layout:
    models/
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

A model is only usable together with its normalizer and config, so runs are always handled
as whole folders (CURRENT names a folder, promote copies the bundle).
"""
import os
import glob
import shutil
import time

from depot import paths

# Files copied into the deployment model folder; SOURCE records where they came from.
BUNDLE_FILES = ("config_used.ini", "best_model.zip", "best_vec_normalize.pkl",
                "depot_ppo_final.zip", "vec_normalize.pkl", "SOURCE")


def new_run_id():
    return time.strftime("run_%Y%m%d-%H%M%S")


def run_dir(run_id):
    return os.path.join(paths.MODELS_DIR, run_id)


def write_current(run_id):
    """Point CURRENT at run_id (the default run for evaluation and the dashboards)."""
    with open(os.path.join(paths.MODELS_DIR, "CURRENT"), "w") as fh:
        fh.write(run_id)


def latest_run_dir():
    found = sorted(glob.glob(os.path.join(paths.MODELS_DIR, "run_*")))
    return found[-1] if found else None


def resolve_run_dir(run=None):
    """Directory of the run to load: explicit `run` (id or path) > CURRENT > newest run.

    Raises FileNotFoundError if no such run exists.
    """
    if run:
        d = run if os.path.isdir(run) else run_dir(run)
    else:
        cur = os.path.join(paths.MODELS_DIR, "CURRENT")
        d = run_dir(open(cur).read().strip()) if os.path.exists(cur) else latest_run_dir()
    if not d or not os.path.isdir(d):
        raise FileNotFoundError(f"no run found ({run or 'CURRENT / latest'}) in {paths.MODELS_DIR}")
    return d


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


def promote(src_run_dir, dest=None):
    """Copy a run's default model, its normalizer and config snapshot into the deployment
    model folder (replacing what was there). Returns the SOURCE label, e.g.
    "run_20261101-041443/best_model"."""
    dest = str(dest or paths.DEPLOY_MODEL_DIR)
    model_path, vn_path = resolve_model(src_run_dir)
    snap = os.path.join(src_run_dir, "config_used.ini")
    if model_path is None or not os.path.exists(snap):
        raise FileNotFoundError(f"no loadable model + config_used.ini in {src_run_dir}")

    os.makedirs(dest, exist_ok=True)
    for name in BUNDLE_FILES:
        old = os.path.join(dest, name)
        if os.path.exists(old):
            os.remove(old)
    for f in (snap, model_path + ".zip", vn_path):
        shutil.copy2(f, os.path.join(dest, os.path.basename(f)))

    source = f"{os.path.basename(os.path.normpath(src_run_dir))}/{os.path.basename(model_path)}"
    with open(os.path.join(dest, "SOURCE"), "w") as fh:
        fh.write(source)
    return source
