"""
Step 1: normalize raw CSV and convert it to SCAGE PKL format.

Example:
  python 1_csv_to_pkl.py --input_csv ./data/mpp/custom/caco2.csv --task caco2 --target_col logPapp
"""

import argparse
import os
import sys
from pathlib import Path

from utils.caco2_pipeline_utils import normalize_csv, resolve_path, run_cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: CSV normalize -> PKL")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, default="./data/mpp/raw")
    parser.add_argument("--pkl_dir", type=str, default="./data/mpp/pkl")
    parser.add_argument("--result_dir", type=str, default="./outputs/caco2_pipeline")
    parser.add_argument("--num_cores", type=int, default=16)
    parser.add_argument("--smiles_timeout_sec", type=int, default=60)
    parser.add_argument("--force_rebuild_pkl", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    py = sys.executable

    raw_dir = resolve_path(project_root, args.raw_dir)
    pkl_dir = resolve_path(project_root, args.pkl_dir)
    result_dir = resolve_path(project_root, args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    task = args.task.lower()
    input_csv = resolve_path(project_root, args.input_csv)
    normalized_csv = raw_dir / f"{task}.csv"
    task_pkl = pkl_dir / f"{task}.pkl"
    failed_csv = result_dir / f"{task}_failed_smiles.csv"

    print(f"[STEP 1A] Normalize CSV -> {normalized_csv}")
    normalize_csv(input_csv, normalized_csv, args.target_col)

    if task_pkl.exists() and not args.force_rebuild_pkl:
        print(f"[STEP 1B] Skip CSV->PKL (exists): {task_pkl}")
    else:
        print("[STEP 1B] CSV -> PKL")
        run_cmd(
            [
                py, "csv_to_pkl.py",
                "--taskname", task,
                "--dataroot", str(raw_dir),
                "--datatarget", str(pkl_dir),
                "--num_cores", str(args.num_cores),
                "--smiles_timeout_sec", str(args.smiles_timeout_sec),
                "--failed_csv_path", str(failed_csv),
            ],
            cwd=project_root,
        )

    print("\n=== Step 1 Completed ===")
    print(f"Normalized CSV: {normalized_csv}")
    print(f"PKL: {task_pkl}")
    if os.path.exists(failed_csv):
        print(f"Failed SMILES CSV: {failed_csv}")


if __name__ == "__main__":
    main()
