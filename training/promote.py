"""Copy a trained run's model bundle into the deployment model folder (deploy/model/).

    python training/promote.py                 # the CURRENT run
    python training/promote.py run_<id>        # a specific run (id or path)

Deployment only ever loads deploy/model/, so a new training run never changes the
deployed model until it is promoted.
"""
import argparse

from depot import paths, runs


def main():
    ap = argparse.ArgumentParser(description="Promote a run to the deployment model folder.")
    ap.add_argument("run", nargs="?", default=None, help="run id / path (default: CURRENT)")
    args = ap.parse_args()

    source = runs.promote(runs.resolve_run_dir(args.run))
    print(f"Promoted {source} -> {paths.DEPLOY_MODEL_DIR}")


if __name__ == "__main__":
    main()
