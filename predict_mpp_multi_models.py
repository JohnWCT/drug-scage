"""
predict_mpp_multi_models.py
===========================
Run multi-model SCAGE MPP inference for a single input CSV.

Key features
------------
  * Input one CSV with a `smiles` column
  * Infer with multiple MPP checkpoints under `weights/mpp`
  * Output one row per valid SMILES and one confidence column per model
  * Model list is configurable via `--models`

Usage Examples
--------------
# 1. Run ALL available models under ./weights/mpp on a custom CSV:
    python predict_mpp_multi_models.py \
        --csv ./data/mpp/custom/Final148_compound_need_ADME.csv \
        --models all \
        --weights_dir ./weights/mpp \
        --output ./result/Final148_multi_confidence.csv

# 2. Run a user-selected list of models:
    python predict_mpp_multi_models.py \
        --csv ./data/mpp/custom/Final148_compound_need_ADME.csv \
        --models bace bbbp clintox \
        --weights_dir ./weights/mpp \
        --output ./result/Final148_selected_models.csv

# 3. Collapse multi-task model outputs (e.g. tox21, sider) into a single
#    column by averaging all task heads:
    python predict_mpp_multi_models.py \
        --csv ./data/mpp/custom/Final148_compound_need_ADME.csv \
        --models all \
        --multihead_reduce mean \
        --output ./result/Final148_mean_confidence.csv

# 4. Tune batch size / workers / GPU device:
    python predict_mpp_multi_models.py \
        --csv ./data/mpp/custom/Final148_compound_need_ADME.csv \
        --models bace bbbp \
        --batch_size 64 \
        --dataloader_num_workers 8 \
        --gpus 0
"""

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _config import get_downstream_task_names, pdir
from data_process.data_collator import collator_finetune_pkl
from predict_mpp import (
    CsvDataset,
    load_model,
    load_precomputed_subset,
    run_inference,
    restore_global_dist_bar_from_ckpt,
)
from datasets.dataloader import FinetuneDataset as FinetuneDataset_pkl
from utils.global_var_util import GlobalVar
from utils.metric_util import compute_cls_metric_tensor_with_pr, compute_reg_metric_with_r2
from utils.userconfig_util import config_current_user, config_dataset_form, drop_last_flag


class DeepCopyDataset(Dataset):
    """
    Wrap a dataset so each ``__getitem__`` returns a deep copy of the item.

    SCAGE's collator (``collator_finetune_pkl``) mutates each item dict in
    place (e.g. pads ``edges``, ``pair_distances`` etc.).  If we iterate the
    same underlying dataset twice (one pass per model), the second pass
    receives already-padded items and crashes inside
    ``padding_pair_distances_loong``.  Deep-copying at read time isolates
    each pass.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return deepcopy(self.base[idx])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-model SCAGE MPP inference on one CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["csv-infer", "pkl-eval"],
        default="csv-infer",
        help="csv-infer: infer all rows in input CSV; pkl-eval: evaluate test split from pkl",
    )
    parser.add_argument("--task", type=str, default="bbbp", help="Task name for pkl-eval mode.")
    parser.add_argument("--dataroot", type=str, default="./data/mpp/pkl",
                        help="PKL directory for pkl-eval mode.")
    parser.add_argument("--splitroot", type=str, default="./data/mpp/split/",
                        help="Split root for pkl-eval mode.")
    parser.add_argument("--split_type", choices=["scaffold", "random_scaffold"], default="scaffold")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Input CSV path. Must contain a `smiles` column.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names to run, e.g. bace bbbp tox21. "
            "Use `all` to load all *.pth under --weights_dir."
        ),
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights/mpp",
        help="Directory that stores model checkpoints (*.pth).",
    )
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default="./result/mpp_multi_confidence.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--multihead_reduce",
        choices=["mean", "max", "first", "all"],
        default="all",
        help=(
            "How to handle multi-task models (e.g. tox21/sider). "
            "`all`: one column per head; otherwise aggregate to one value."
        ),
    )
    return parser.parse_args()


def resolve_models(weights_dir: str, models_arg: List[str]) -> List[str]:
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights_dir not found: {weights_dir}")

    available = sorted([p.stem for p in weights_path.glob("*.pth")])
    if not available:
        raise FileNotFoundError(f"No .pth checkpoints found under: {weights_dir}")

    if len(models_arg) == 1 and models_arg[0].lower() == "all":
        return available

    missing = [m for m in models_arg if m not in available]
    if missing:
        raise ValueError(
            f"Requested models not found in {weights_dir}: {missing}. "
            f"Available: {available}"
        )
    return models_arg


def build_runtime_config(base_cfg: Dict, task_name: str, ckpt_path: str, args) -> Dict:
    cfg = deepcopy(base_cfg)
    user = "mpp"
    cfg["userconfig"][user]["dataset_dir"] = args.dataroot
    cfg["userconfig"][user]["split_dir"] = args.splitroot
    cfg = config_current_user(user, cfg)
    cfg = config_dataset_form("pkl", cfg)
    cfg["task_name"] = task_name
    cfg["split_type"] = args.split_type
    cfg["seed"] = args.seed
    cfg["batch_size"] = args.batch_size
    cfg["dataloader_num_workers"] = args.dataloader_num_workers
    cfg["checkpoint"] = ckpt_path
    cfg = get_downstream_task_names(cfg)

    GlobalVar.dist_bar = [0, 0]
    GlobalVar.freeze_layers = 0
    GlobalVar.parallel_train = False
    GlobalVar.use_ckpt = True
    cfg["freeze_layers"] = 0
    return cfg


def print_eval_metrics(task_type: str, raw_preds: np.ndarray, labels: np.ndarray):
    y_true = torch.tensor(labels.reshape(len(labels), -1), device='cuda')
    y_pred = torch.tensor(raw_preds.reshape(len(raw_preds), -1), device='cuda')
    if task_type == "classification":
        roc_auc, pr_auc = compute_cls_metric_tensor_with_pr(y_true, y_pred)
        print(f"  test_auc: {roc_auc:.6f} | test_pr_auc: {pr_auc:.6f}")
    else:
        mae, rmse, r2 = compute_reg_metric_with_r2(y_true, y_pred)
        print(f"  test_mae: {mae:.6f} | test_rmse: {rmse:.6f} | test_r2: {r2:.6f}")


def to_model_columns(model_name: str, preds: np.ndarray, target_cols: List[str], reduce_mode: str):
    preds_2d = preds.reshape(len(preds), -1)
    if preds_2d.shape[1] == 1:
        return {model_name: preds_2d[:, 0]}

    if reduce_mode == "all":
        return {
            f"{model_name}__{target_cols[i]}": preds_2d[:, i]
            for i in range(preds_2d.shape[1])
        }
    if reduce_mode == "mean":
        return {model_name: preds_2d.mean(axis=1)}
    if reduce_mode == "max":
        return {model_name: preds_2d.max(axis=1)}
    if reduce_mode == "first":
        return {model_name: preds_2d[:, 0]}
    raise ValueError(f"Unknown reduce mode: {reduce_mode}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    cfg_path = Path(pdir) / "config" / "config_finetune.yaml"
    base_cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.FullLoader)

    models = resolve_models(args.weights_dir, args.models)
    print(f"Models to run ({len(models)}): {models}")

    if args.mode == "csv-infer":
        if not args.csv:
            raise ValueError("--csv is required when --mode=csv-infer")
        csv_dataset = CsvDataset(
            csv_path=args.csv,
            target_cols=None,
            num_workers=args.dataloader_num_workers,
        )
        if len(csv_dataset) == 0:
            raise RuntimeError("No valid molecules after featurisation.")
        safe_dataset = DeepCopyDataset(csv_dataset)
        loader = DataLoader(
            safe_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator_finetune_pkl,
            drop_last=False,
        )
        sample_ids = csv_dataset.original_indices
        smiles_source = csv_dataset.smiles_list
        has_labels = False
        labels_ref = None
    else:
        dataset = FinetuneDataset_pkl(root=args.dataroot, task_name=args.task)
        test_dataset = load_precomputed_subset(
            dataset=dataset,
            splitroot=args.splitroot,
            split_type=args.split_type,
            task_name=args.task,
            seed=args.seed,
            split_part="test",
        )
        loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collator_finetune_pkl,
            drop_last=drop_last_flag(len(test_dataset), args.batch_size),
        )
        sample_ids = list(test_dataset.indices)
        smiles_source = [dataset.data[i]["smiles"] for i in test_dataset.indices]
        has_labels = True
        labels_ref = None

    out_df = pd.DataFrame({"sample_id": sample_ids, "smiles": smiles_source})

    for model_name in tqdm(models, desc="Running models"):
        ckpt = str(Path(args.weights_dir) / f"{model_name}.pth")
        runtime_cfg = build_runtime_config(base_cfg, model_name, ckpt, args)
        model = load_model(runtime_cfg)
        restore_global_dist_bar_from_ckpt(ckpt)
        _, preds, labels, raw_preds = run_inference(
            model=model,
            loader=loader,
            smiles_source=smiles_source,
            has_labels=has_labels,
            task_type=runtime_cfg["task"],
            return_raw=True,
        )
        if has_labels and labels_ref is None:
            labels_ref = labels

        new_cols = to_model_columns(
            model_name=model_name,
            preds=preds,
            target_cols=runtime_cfg["target"],
            reduce_mode=args.multihead_reduce,
        )
        for col_name, values in new_cols.items():
            out_df[col_name] = np.round(values.astype(float), 6)
        if has_labels:
            print(f"[{model_name}]")
            print_eval_metrics(runtime_cfg["task"], raw_preds, labels)

        del model
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved multi-model confidence CSV: {output_path} ({len(out_df)} rows)")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
