"""
Step 2: build train/valid/test split for the normalized CSV.

Example:
  python 2_data_split.py --task caco2 --split_type random_scaffold --seed 8
"""

import argparse
from pathlib import Path

import pandas as pd

from utils.caco2_pipeline_utils import (
    build_cyclic_scaffold_split,
    build_random_scaffold_split,
    build_scaffold_split,
    get_split_path,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: create data split")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--split_type",
        type=str,
        default="random_scaffold",
        choices=["scaffold", "random_scaffold", "cyclic_scaffold"],
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--raw_dir", type=str, default="./data/mpp/raw")
    parser.add_argument("--split_root", type=str, default="./data/mpp")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--output_split_pkl", type=str, default=None)
    parser.add_argument("--force_rebuild_split", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    task = args.task.lower()

    raw_dir = resolve_path(project_root, args.raw_dir)
    split_root = resolve_path(project_root, args.split_root)
    normalized_csv = raw_dir / f"{task}.csv"
    if not normalized_csv.exists():
        raise FileNotFoundError(f"找不到 normalized CSV，請先執行 1_csv_to_pkl.py: {normalized_csv}")

    if args.output_split_pkl:
        split_pkl = resolve_path(project_root, args.output_split_pkl)
    else:
        split_pkl = get_split_path(
            split_root,
            task,
            args.split_type,
            args.seed,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
        )
    if split_pkl.exists() and not args.force_rebuild_split:
        print(f"[STEP 2] Skip split build (exists): {split_pkl}")
    else:
        df = pd.read_csv(normalized_csv)
        if "smiles" not in df.columns:
            raise ValueError(f"normalized CSV 必須包含 smiles 欄位: {normalized_csv}")
        print(f"[STEP 2] Build {args.split_type} split")
        if args.split_type == "scaffold":
            split_pkl = build_scaffold_split(task, split_root, df["smiles"].tolist())
        elif args.split_type == "random_scaffold":
            split_pkl = build_random_scaffold_split(
                task,
                split_root,
                df["smiles"].tolist(),
                args.seed,
            )
        elif args.split_type == "cyclic_scaffold":
            split_pkl = build_cyclic_scaffold_split(
                task,
                split_root,
                df["smiles"].tolist(),
                args.seed,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                output_split_pkl=split_pkl,
            )
        else:
            raise ValueError(f"Unsupported split_type: {args.split_type}")

    print("\n=== Step 2 Completed ===")
    print(f"Split: {split_pkl}")


if __name__ == "__main__":
    main()
