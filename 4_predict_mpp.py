"""
Step 4: evaluate the trained model and export a PKL-based prediction table.

Example:
  python 4_predict_mpp.py --task caco2 --split_type random_scaffold --seed 8 --ckpt ./outputs/caco2_pipeline/caco2_best.pth
"""

import argparse
import sys
from pathlib import Path

from utils.caco2_pipeline_utils import (
    resolve_path,
    run_cmd,
    save_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: evaluate and export predictions")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split_type", type=str, default="random_scaffold", choices=["scaffold", "random_scaffold"])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--ckpt", type=str, default=None, help="覆寫要評估的模型路徑")
    parser.add_argument("--pkl_dir", type=str, default="./data/mpp/pkl")
    parser.add_argument("--split_root", type=str, default="./data/mpp")
    parser.add_argument("--result_dir", type=str, default="./outputs/caco2_pipeline")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    py = sys.executable
    task = args.task.lower()
    settings = {
        "task": task,
        "split_type": args.split_type,
        "seed": args.seed,
        "gpus": args.gpus,
        "pkl_dir": args.pkl_dir,
        "split_root": args.split_root,
        "result_dir": args.result_dir,
        "batch_size": args.batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
    }

    pkl_dir = resolve_path(project_root, args.pkl_dir)
    result_dir = resolve_path(project_root, args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    final_weights = resolve_path(project_root, args.ckpt) if args.ckpt else project_root / "weights" / "mpp" / f"{task}.pth"
    if not final_weights.exists():
        raise FileNotFoundError(f"找不到模型權重，請先執行 3_finetune_2step.py: {final_weights}")

    print("[STEP 4A] Evaluate train/valid/test/all_data")
    _, score_json = save_scores(py, project_root, settings, final_weights)

    print("[STEP 4B] Export PKL-based per-sample prediction vs ground truth")
    pred_csv = result_dir / f"{task}_predictions_with_gt.csv"
    run_cmd(
        [
            py, "predict_mpp.py",
            "--mode", "pkl-infer",
            "--task", task,
            "--dataroot", str(pkl_dir),
            "--ckpt", str(final_weights),
            "--batch_size", str(args.batch_size),
            "--dataloader_num_workers", str(args.dataloader_num_workers),
            "--gpus", str(args.gpus),
            "--output", str(pred_csv),
        ],
        cwd=project_root,
    )

    print("\n=== Step 4 Completed ===")
    print(f"Weights: {final_weights}")
    print(f"Score JSON: {score_json}")
    print(f"Predictions CSV: {pred_csv}")


if __name__ == "__main__":
    main()
