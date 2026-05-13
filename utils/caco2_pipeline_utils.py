import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from _config import task_configs
from prepare_data import ScaffoldSplitter, generate_scaffold


def run_cmd(cmd, cwd, print_cmd=True):
    if print_cmd:
        print(f"[RUN] {' '.join(str(x) for x in cmd)}")
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    merged_output = bytearray()
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1024)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        merged_output.extend(chunk)
    proc.wait()
    if proc.returncode != 0:
        output_text = merged_output.decode("utf-8", errors="replace").strip()
        if len(output_text) > 4000:
            output_text = output_text[-4000:]
        detail = f"\n--- command output tail ---\n{output_text}" if output_text else ""
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(str(x) for x in cmd)}{detail}"
        )
    return merged_output.decode("utf-8", errors="replace")


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _override(args, name, default=None):
    return getattr(args, name, default) if args is not None else default


def load_pipeline_settings(project_root: Path, args=None):
    config_arg = _override(args, "pipeline_config", "./config/config_caco2_pipeline.yaml")
    cfg_path = Path(config_arg)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到 pipeline 設定檔: {cfg_path}")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.FullLoader) or {}
    optim_cfg = cfg.get("optim", {}) or {}
    scheduler_cfg = cfg.get("lr_scheduler", {}) or {}
    two_stage_cfg = cfg.get("two_stage", {}) or {}
    search_cfg = cfg.get("search", {}) or {}

    task = _override(args, "task") or cfg.get("task_name")
    if task is None:
        raise ValueError("請在 YAML 設定 task_name，或使用 --task 指定 task")
    task = task.lower()
    if task not in task_configs:
        raise ValueError(f"task 不支援: {task}，可用 tasks: {sorted(task_configs.keys())}")

    settings = {
        "config_path": str(cfg_path),
        "input_csv": _override(args, "input_csv") or cfg.get("input_csv", "./data/mpp/custom/caco2.csv"),
        "task": task,
        "target_col": _override(args, "target_col") or cfg.get("target_col") or task_configs[task][0][0],
        "raw_dir": cfg.get("raw_dir", "./data/mpp/raw"),
        "pkl_dir": cfg.get("pkl_dir", "./data/mpp/pkl"),
        "split_root": cfg.get("split_root", "./data/mpp"),
        "result_dir": cfg.get("result_dir", "./outputs/caco2_pipeline"),
        "split_type": _override(args, "split_type") or cfg.get("split_type", "scaffold"),
        "seed": int(_override(args, "seed", cfg.get("seed", 8))),
        "gpus": str(_override(args, "gpus") or cfg.get("gpus", "0")),
        "num_cores": int(cfg.get("num_cores", 16)),
        "smiles_timeout_sec": int(cfg.get("smiles_timeout_sec", 60)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 4)),
        "pin_memory": bool(cfg.get("pin_memory", False)),
        "batch_size": int(cfg.get("batch_size", 32)),
        "epochs": int(cfg.get("epochs", 100)),
        "patience": int(cfg.get("patience", 30)),
        "save_ckpt": int(cfg.get("save_ckpt", 10)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "ckpt": str(cfg.get("ckpt", "pretrain")),
        "lr": float(optim_cfg.get("init_lr", 5e-5)),
        "weight_decay": float(optim_cfg.get("weight_decay", 1e-4)),
        "optim_type": str(optim_cfg.get("type", "adam")),
        "scheduler_type": str(scheduler_cfg.get("type", "None")),
        "warm_up_epoch": int(scheduler_cfg.get("warm_up_epoch", 5)),
        "two_stage": two_stage_cfg,
        "search": search_cfg,
    }
    if settings["split_type"] not in {"scaffold", "random_scaffold", "cyclic_scaffold"}:
        raise ValueError(f"Unsupported split_type: {settings['split_type']}")
    print(f"[INFO] Pipeline config loaded: {cfg_path}")
    return settings


def settings_namespace(settings):
    return SimpleNamespace(
        task=settings["task"],
        split_type=settings["split_type"],
        seed=settings["seed"],
        batch_size=settings["batch_size"],
        num_workers=settings["dataloader_num_workers"],
        gpus=settings["gpus"],
        result_dir=settings["result_dir"],
        split_pkl_path=settings.get("split_pkl_path"),
        allow_empty_test=bool(settings.get("allow_empty_test", False)),
    )


def normalize_csv(input_csv: Path, output_csv: Path, target_col: str):
    df = pd.read_csv(input_csv)
    smiles_col = None
    for candidate in ["smiles", "SMILES"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        raise ValueError("CSV 必須包含 smiles 或 SMILES 欄位")
    if target_col not in df.columns:
        raise ValueError(f"CSV 缺少目標欄位: {target_col}")

    out_df = df[[smiles_col, target_col]].copy()
    out_df = out_df.rename(columns={smiles_col: "smiles"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def build_scaffold_split(task_name: str, split_root: Path, smiles_list):
    split_scaffold_dir = split_root / "split" / "scaffold"
    split_scaffold_dir.mkdir(parents=True, exist_ok=True)
    ScaffoldSplitter(task_name, str(split_root), list(smiles_list))
    return split_scaffold_dir / f"{task_name}.pkl"


def build_random_scaffold_split(task_name: str, split_root: Path, smiles_list, seed: int):
    rng = np.random.RandomState(seed)
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    n_total = len(smiles_list)
    all_indices = rng.permutation(n_total).tolist()
    n_train = int(frac_train * n_total)
    n_valid = int(frac_valid * n_total)
    split_dict = {
        "train_idx": sorted(all_indices[:n_train]),
        "valid_idx": sorted(all_indices[n_train:n_train + n_valid]),
        "test_idx": sorted(all_indices[n_train + n_valid:]),
    }
    split_random_dir = split_root / "split" / "random_scaffold"
    split_random_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_random_dir / f"{task_name}_{seed}.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(split_dict, f)
    return split_path


def build_cyclic_scaffold_split(
    task_name: str,
    split_root: Path,
    smiles_list,
    seed: int,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_split_pkl: Path = None,
    chunk_size: int = 10,
):
    if valid_ratio <= 0:
        raise ValueError("valid_ratio must be > 0 because model selection needs validation set.")
    if test_ratio < 0:
        raise ValueError("test_ratio must be >= 0.")
    if chunk_size <= 1:
        raise ValueError("chunk_size must be > 1.")

    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices[generate_scaffold(smiles, include_chirality=False)].append(idx)

    grouped_by_size = defaultdict(list)
    for scaffold_set in scaffold_to_indices.values():
        grouped_by_size[len(scaffold_set)].append(sorted(scaffold_set))

    rng = random.Random(seed)
    all_scaffold_sets = []
    for size in sorted(grouped_by_size.keys(), reverse=True):
        group = grouped_by_size[size]
        rng.shuffle(group)
        all_scaffold_sets.extend(group)

    train_idx, valid_idx, test_idx = [], [], []
    for chunk_index, start in enumerate(range(0, len(all_scaffold_sets), chunk_size)):
        chunk = list(all_scaffold_sets[start:start + chunk_size])
        if not chunk:
            continue
        chunk_rng = random.Random(seed + chunk_index)
        chunk_rng.shuffle(chunk)
        if test_ratio == 0:
            valid_sets = chunk[:1]
            test_sets = []
            train_sets = chunk[1:]
        else:
            valid_sets = chunk[:1]
            test_sets = chunk[1:2]
            train_sets = chunk[2:]

        for scaffold_set in train_sets:
            train_idx.extend(scaffold_set)
        for scaffold_set in valid_sets:
            valid_idx.extend(scaffold_set)
        for scaffold_set in test_sets:
            test_idx.extend(scaffold_set)

    if test_ratio == 0:
        test_idx = []

    n_total = len(smiles_list)
    split_dict = {
        "train_idx": sorted(train_idx),
        "valid_idx": sorted(valid_idx),
        "test_idx": sorted(test_idx),
        "meta": {
            "task": task_name,
            "split_type": "cyclic_scaffold",
            "seed": seed,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "chunk_size": chunk_size,
            "n_total": n_total,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_test": len(test_idx),
            "actual_val_ratio": len(valid_idx) / n_total if n_total else 0.0,
            "actual_test_ratio": len(test_idx) / n_total if n_total else 0.0,
            "assignment_rule": (
                "chunk_size=10; shuffle each chunk by seed+chunk_index; "
                "test_ratio=0 -> 1 valid + rest train; "
                "test_ratio>0 -> 1 valid + 1 test + rest train"
            ),
        },
    }

    if output_split_pkl is None:
        output_split_pkl = get_split_path(
            split_root,
            task_name,
            "cyclic_scaffold",
            seed,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
    output_split_pkl = Path(output_split_pkl)
    output_split_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_split_pkl, "wb") as f:
        pickle.dump(split_dict, f)

    print("[STEP 2] cyclic scaffold split summary")
    print(split_dict["meta"])
    print(f"[STEP 2] Saved split: {output_split_pkl}")
    return output_split_pkl


def get_split_path(
    split_root: Path,
    task_name: str,
    split_type: str,
    seed: int,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if split_type == "scaffold":
        return split_root / "split" / "scaffold" / f"{task_name}.pkl"
    if split_type == "random_scaffold":
        return split_root / "split" / "random_scaffold" / f"{task_name}_{seed}.pkl"
    if split_type == "cyclic_scaffold":
        return (
            split_root
            / "split"
            / "cyclic_scaffold"
            / f"{task_name}_cyclic_scaffold_seed{seed}_v{valid_ratio:.2f}_t{test_ratio:.2f}.pkl"
        )
    raise ValueError(f"Unsupported split_type: {split_type}")


def find_latest_model(project_root: Path, run_tag: str = None):
    candidates = []
    search_roots = [project_root / "finetune_result", project_root.parent / "train_result" / "finetune_result"]
    for root in search_roots:
        if not root.exists():
            continue
        patterns = ["model.pth", "model_*.pth", "checkpoint/model_*.pth"]
        for pattern in patterns:
            for path in root.rglob(pattern):
                if run_tag is None or run_tag in str(path.parent):
                    candidates.append(path)

    if not candidates:
        suffix = f" for run_tag={run_tag}" if run_tag else ""
        raise FileNotFoundError(f"找不到 finetune 產生的模型檔{suffix}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_predict_metrics(output_text: str):
    mae = re.findall(r"test_mae:\s*([0-9eE.+-]+)", output_text)
    rmse = re.findall(r"test_rmse:\s*([0-9eE.+-]+)", output_text)
    r2 = re.findall(r"test_r2:\s*([0-9eE.+-]+)", output_text)
    auc = re.findall(r"test_auc:\s*([0-9eE.+-]+)", output_text)
    pr_auc = re.findall(r"test_pr_auc:\s*([0-9eE.+-]+)", output_text)
    if rmse:
        return {
            "task_type": "regression",
            "primary_metric": "rmse",
            "mae": float(mae[-1]) if mae else None,
            "rmse": float(rmse[-1]),
            "r2": float(r2[-1]) if r2 else None,
        }
    if auc:
        return {
            "task_type": "classification",
            "primary_metric": "roc_auc",
            "roc_auc": float(auc[-1]),
            "pr_auc": float(pr_auc[-1]) if pr_auc else None,
        }
    return {"task_type": "unknown", "primary_metric": None}


def run_predict_eval_metrics_only(py, project_root: Path, args, pkl_dir: Path, split_root: Path,
                                  final_weights: Path, split_name: str):
    if split_name == "test" and getattr(args, "allow_empty_test", False):
        return {
            "task_type": "regression",
            "primary_metric": "rmse",
            "mae": None,
            "rmse": None,
            "r2": None,
        }
    result_dir = (project_root / args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_output = tempfile.mkstemp(prefix=f"{args.task}_{split_name}_", suffix=".csv", dir=str(result_dir))
    os.close(fd)
    try:
        cmd = [
            py, "predict_mpp.py",
            "--mode", "eval",
            "--task", args.task,
            "--dataroot", str(pkl_dir),
            "--splitroot", str(split_root / "split"),
            "--split_type", args.split_type,
            "--split_seed", str(args.seed),
            "--eval_split", split_name,
            "--ckpt", str(final_weights),
            "--batch_size", str(args.batch_size),
            "--dataloader_num_workers", str(args.num_workers),
            "--gpus", str(args.gpus),
            "--output", str(tmp_output),
        ]
        if getattr(args, "split_pkl_path", None):
            cmd.extend(["--split_pkl_path", str(args.split_pkl_path)])
        out = run_cmd(
            cmd,
            cwd=project_root,
            print_cmd=False,
        )
        return parse_predict_metrics(out)
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def run_predict_all_metrics_only(py, project_root: Path, args, pkl_dir: Path, final_weights: Path):
    result_dir = (project_root / args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_output = tempfile.mkstemp(prefix=f"{args.task}_all_", suffix=".csv", dir=str(result_dir))
    os.close(fd)
    try:
        out = run_cmd(
            [
                py, "predict_mpp.py",
                "--mode", "pkl-infer",
                "--task", args.task,
                "--dataroot", str(pkl_dir),
                "--ckpt", str(final_weights),
                "--batch_size", str(args.batch_size),
                "--dataloader_num_workers", str(args.num_workers),
                "--gpus", str(args.gpus),
                "--output", str(tmp_output),
            ],
            cwd=project_root,
            print_cmd=False,
        )
        return parse_predict_metrics(out)
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


def save_scores(py, project_root: Path, settings, final_weights: Path):
    pkl_dir = resolve_path(project_root, settings["pkl_dir"])
    split_root = resolve_path(project_root, settings["split_root"])
    result_dir = resolve_path(project_root, settings["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    args = settings_namespace(settings)
    metrics = {
        "train": run_predict_eval_metrics_only(py, project_root, args, pkl_dir, split_root, final_weights, "train"),
        "valid": run_predict_eval_metrics_only(py, project_root, args, pkl_dir, split_root, final_weights, "valid"),
        "test": run_predict_eval_metrics_only(py, project_root, args, pkl_dir, split_root, final_weights, "test"),
        "all_data": run_predict_all_metrics_only(py, project_root, args, pkl_dir, final_weights),
    }
    score_json = result_dir / f"{settings['task']}_score.json"
    with open(score_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Score saved: {score_json}")
    return metrics, score_json


def copy_model_to_result_dir(result_dir: Path, task: str, source_model: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    final_weights = result_dir / f"{task}_best.pth"
    shutil.copy2(source_model, final_weights)
    print(f"[INFO] Best model copied: {final_weights}")
    return final_weights
