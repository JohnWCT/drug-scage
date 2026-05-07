"""
Offline aggregate for brute-force MPP finetune search.

Scans finetune_result/*/trial_params.yaml, detects completed checkpoints, optionally
re-runs predict_mpp eval (same as the main pipeline), and writes *_partial.csv tables.

Docker（建議在容器內執行，勿在本機改動映像外環境）
----------------------------------------------------------------
在專案根目錄啟動容器（本機只負責掛載，不在本機安裝 CUDA / Python 套件）::

  docker run --gpus all --name scage -itd -p 8888:8888 -v \"$(pwd):/workspace\" scage:cu121
  docker exec -it scage bash
  cd /workspace

專案路徑在容器內為 /workspace/；訓練產物（finetune_result、outputs）一併落在掛載目錄，
本機與容器看到的檔案相同。路徑一律以「實際掃到的 trial 目錄」組出 checkpoint，不依賴
trial_params.yaml 內舊的絕對路徑（例如曾寫成 /workspace/... 或別台機器路徑）。

範例（容器內）::

  python 3.1_finetune_search_aggregate.py \\
    --search_config ./config/caco2_finetune_search.yaml \\
    --result_dir ./outputs/caco2_pipeline \\
    --project_root /workspace

若專案不在 /workspace 根目錄，請將 --project_root 設為含 SCAGE 程式與 finetune_result 的目錄。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from utils.caco2_pipeline_utils import resolve_path, run_predict_eval_metrics_only, settings_namespace
from utils.mpp_finetune_report import (
    build_candidate_row,
    build_stage_score_rows,
    primary_valid_score_for_ranking,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate finetune search trials into partial CSV tables")
    p.add_argument("--search_config", type=str, default=None, help="YAML/JSON with task, dataroot, ...")
    p.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Repo root containing finetune_result/ (default: directory of this script). In Docker use /workspace if the repo is mounted there.",
    )
    p.add_argument(
        "--finetune_glob",
        type=str,
        default="finetune_result/*/trial_params.yaml",
        help="Glob pattern relative to project_root for trial_params.yaml files",
    )
    p.add_argument("--result_dir", type=str, default="./outputs/caco2_pipeline")
    p.add_argument("--task", type=str, default="caco2")
    p.add_argument("--dataroot", type=str, default="./data/mpp/pkl")
    p.add_argument("--splitroot", type=str, default="./data/mpp/split")
    p.add_argument("--split_type", type=str, default="random_scaffold")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Include two-stage trials that finished stage1 only (stage2/model.pth missing)",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only list trials and hyperparameters; do not run predict_mpp (metrics empty)",
    )
    p.add_argument("--also-json", action="store_true", help="Write a compact *_aggregate_partial.json")
    return p.parse_args()


def read_search_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        if cfg_path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def eval_settings_namespace(args: argparse.Namespace, trial: Dict[str, Any]):
    splitroot = Path(args.splitroot)
    split_root = splitroot.parent if splitroot.name == "split" else splitroot.parent
    settings = {
        "task": args.task,
        "split_type": args.split_type,
        "seed": trial["seed"],
        "batch_size": trial["batch_size"],
        "dataloader_num_workers": args.dataloader_num_workers,
        "gpus": args.gpus,
        "result_dir": args.result_dir,
        "pkl_dir": args.dataroot,
        "split_root": str(split_root),
    }
    return settings_namespace(settings)


def eval_pair(
    py: str,
    project_root: Path,
    eval_ns,
    pkl_dir: Path,
    split_root: Path,
    ckpt: Optional[Path],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    if ckpt is None or not ckpt.is_file():
        return None, None
    valid_m = run_predict_eval_metrics_only(py, project_root, eval_ns, pkl_dir, split_root, ckpt, "valid")
    test_m = run_predict_eval_metrics_only(py, project_root, eval_ns, pkl_dir, split_root, ckpt, "test")
    return valid_m, test_m


def ranking_scalar(candidate: Dict[str, Any]) -> Optional[float]:
    s2 = primary_valid_score_for_ranking(candidate.get("valid"))
    if s2 is not None:
        return s2
    return primary_valid_score_for_ranking(candidate.get("stage1_valid"))


def build_one_candidate(
    payload: Dict[str, Any],
    trial_root: Path,
    args: argparse.Namespace,
    py: str,
    project_root: Path,
    pkl_dir: Path,
    split_root: Path,
    include_partial: bool,
    skip_eval: bool,
) -> Optional[Dict[str, Any]]:
    search_trial: Dict[str, Any] = payload.get("search_trial") or {}
    stages: Dict[str, Any] = payload.get("stages") or {}
    idx = int(payload.get("trial_serial") or search_trial.get("trial_serial") or 0)

    single_stage = "single" in stages
    eval_ns = eval_settings_namespace(args, search_trial)

    if single_stage:
        ckpt_single = trial_root / "single" / "model.pth"
        if not ckpt_single.is_file():
            return None
        valid_m = test_m = None
        if not skip_eval:
            valid_m, test_m = eval_pair(py, project_root, eval_ns, pkl_dir, split_root, ckpt_single)
        return {
            "idx": idx,
            "trial_dir": str(trial_root.resolve()),
            "run_tag": "single",
            "stage1_run_tag": None,
            "stage2_run_tag": None,
            "stage1_model": None,
            "stage2_model": None,
            "model": str(ckpt_single.resolve()),
            "final_weights": None,
            "params": {**search_trial, "single_stage": True},
            "stage1_valid": None,
            "stage1_test": None,
            "valid": valid_m,
            "test": test_m,
            "completion": "full",
        }

    p1 = trial_root / "stage1" / "model.pth"
    p2 = trial_root / "stage2" / "model.pth"

    if p2.is_file():
        s1v = s1t = v = t = None
        if not skip_eval:
            s1v, s1t = eval_pair(py, project_root, eval_ns, pkl_dir, split_root, p1)
            v, t = eval_pair(py, project_root, eval_ns, pkl_dir, split_root, p2)
        return {
            "idx": idx,
            "trial_dir": str(trial_root.resolve()),
            "run_tag": "stage2",
            "stage1_run_tag": "stage1",
            "stage2_run_tag": "stage2",
            "stage1_model": str(p1.resolve()) if p1.is_file() else None,
            "stage2_model": str(p2.resolve()),
            "model": str(p2.resolve()),
            "final_weights": None,
            "params": {**search_trial, "single_stage": False},
            "stage1_valid": s1v,
            "stage1_test": s1t,
            "valid": v,
            "test": t,
            "completion": "full",
        }

    if include_partial and p1.is_file():
        s1v = s1t = None
        if not skip_eval:
            s1v, s1t = eval_pair(py, project_root, eval_ns, pkl_dir, split_root, p1)
        return {
            "idx": idx,
            "trial_dir": str(trial_root.resolve()),
            "run_tag": "stage1",
            "stage1_run_tag": "stage1",
            "stage2_run_tag": "stage2",
            "stage1_model": str(p1.resolve()),
            "stage2_model": None,
            "model": str(p1.resolve()),
            "final_weights": None,
            "params": {**search_trial, "single_stage": False},
            "stage1_valid": s1v,
            "stage1_test": s1t,
            "valid": None,
            "test": None,
            "completion": "stage1_only",
        }

    return None


def _to_jsonable(obj: Any) -> Any:
    """Make nested dicts safe for json.dump (numpy scalars, Path, etc.)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
    except ImportError:
        pass
    if hasattr(obj, "item"):
        try:
            return _to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def mark_best_among_scanned(candidates: List[Dict[str, Any]]) -> None:
    scored: List[Tuple[float, int]] = []
    for i, c in enumerate(candidates):
        r = ranking_scalar(c)
        if r is not None:
            scored.append((r, i))
    if not scored:
        for c in candidates:
            c["is_best"] = False
            c["is_best_among_scanned"] = False
        return
    # Higher score wins; tie-break by lower trial idx (deterministic).
    scored.sort(key=lambda x: (-x[0], candidates[x[1]]["idx"]))
    best_i = scored[0][1]
    for j, c in enumerate(candidates):
        c["is_best"] = False
        c["is_best_among_scanned"] = j == best_i


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    py = sys.executable

    file_cfg = read_search_config(args.search_config)
    for key in ["task", "dataroot", "splitroot", "split_type", "result_dir", "gpus", "dist_bar"]:
        if key in file_cfg:
            setattr(args, key, file_cfg[key])

    pkl_dir = resolve_path(project_root, args.dataroot)
    splitroot = resolve_path(project_root, args.splitroot)
    split_root = splitroot.parent if splitroot.name == "split" else splitroot.parent
    result_dir = resolve_path(project_root, args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    trial_yaml_paths = sorted(project_root.glob(args.finetune_glob))
    if not trial_yaml_paths:
        print(f"[WARN] No files matched under {project_root}: {args.finetune_glob!r}", flush=True)
        sys.exit(1)

    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for yaml_path in trial_yaml_paths:
        trial_root = yaml_path.parent
        with open(yaml_path, "r", encoding="utf-8") as f:
            payload = yaml.load(f, Loader=yaml.FullLoader) or {}

        cand = build_one_candidate(
            payload,
            trial_root,
            args,
            py,
            project_root,
            pkl_dir,
            split_root,
            args.include_partial,
            args.skip_eval,
        )
        if cand is None:
            skipped += 1
            continue
        candidates.append(cand)

    candidates.sort(key=lambda c: c["idx"])
    mark_best_among_scanned(candidates)

    wide_rows = [build_candidate_row(c) for c in candidates]
    stage_rows: List[Dict[str, Any]] = []
    for c in candidates:
        stage_rows.extend(build_stage_score_rows(c))

    task = args.task
    csv_wide = result_dir / f"{task}_finetune_search_candidates_partial.csv"
    csv_stage = result_dir / f"{task}_finetune_2step_score_table_partial.csv"

    pd.DataFrame(wide_rows).to_csv(csv_wide, index=False)
    pd.DataFrame(stage_rows).to_csv(csv_stage, index=False)

    print(f"[INFO] Wrote {len(wide_rows)} candidate rows (skipped incomplete: {skipped})", flush=True)
    print(f"[INFO] Wide table:  {csv_wide}", flush=True)
    print(f"[INFO] Stage table: {csv_stage}", flush=True)

    if args.also_json:
        json_path = result_dir / f"{task}_finetune_aggregate_partial.json"
        best_idx = next((c["idx"] for c in candidates if c.get("is_best_among_scanned")), None)
        payload_out = {
            "task": task,
            "project_root": str(project_root),
            "trials_yaml_matched": len(trial_yaml_paths),
            "skipped_incomplete": skipped,
            "candidates_written": len(candidates),
            "best_idx_among_scanned": best_idx,
            "outputs": {"candidates_csv": str(csv_wide), "stage_csv": str(csv_stage)},
            "candidates": _to_jsonable(candidates),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload_out, f, indent=2, ensure_ascii=False)
        print(f"[INFO] JSON snapshot: {json_path}", flush=True)


if __name__ == "__main__":
    main()
