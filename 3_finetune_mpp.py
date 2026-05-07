"""
Step 3: run single-step MPP finetuning.

This wrapper only launches the upstream finetune_mpp.py script.
It does not run the custom two-step trainer or JSON/YAML grid search.

Example:
  python 3_finetune_2step.py \
    --task caco2 \
    --dataroot ./data/mpp/pkl \
    --splitroot ./data/mpp/split \
    --split_type random_scaffold \
    --seed 8 \
    --epochs 120 \
    --batch_size 16 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --dropout 0.1 \
    --patience 25 \
    --gpus 0
"""

import argparse
import sys
from pathlib import Path

from utils.caco2_pipeline_utils import run_cmd


UPSTREAM_FINETUNE = "finetune_mpp.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3 wrapper: launch upstream finetune_mpp.py single-step MPP finetuning"
    )
    parser.add_argument("--task", type=str, default="caco2", help="MPP task name")
    parser.add_argument("--dataroot", type=str, default="./data/mpp/pkl", help="PKL dataset directory")
    parser.add_argument("--splitroot", type=str, default="./data/mpp/split", help="Split directory")
    parser.add_argument("--split_type", type=str, default="random_scaffold",
                        choices=["scaffold", "random_scaffold"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Forwarded to finetune_mpp.py as --emdedding_dim")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layer_num", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--optim_type", type=str, default="adam", choices=["adam", "rms", "sgd"])
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save_ckpt", type=int, default=1)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--dist_bar", nargs="+", type=int, default=[20, 100])
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--ckpt", type=str, default="pretrain",
                        help="Pretrain checkpoint name used by finetune_mpp.py, e.g. pretrain or None")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    py = sys.executable

    cmd = [
        py, UPSTREAM_FINETUNE,
        "--task", args.task,
        "--dataroot", args.dataroot,
        "--splitroot", args.splitroot,
        "--split_type", args.split_type,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--pin_memory", str(args.pin_memory),
        "--emdedding_dim", str(args.embedding_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--layer_num", str(args.layer_num),
        "--num_heads", str(args.num_heads),
        "--optim_type", args.optim_type,
        "--patience", str(args.patience),
        "--save_ckpt", str(args.save_ckpt),
        "--warm_up_epoch", str(args.warm_up_epoch),
        "--dist_bar", *[str(x) for x in args.dist_bar],
        "--gpus", args.gpus,
        "--ckpt", args.ckpt,
    ]

    run_cmd(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
