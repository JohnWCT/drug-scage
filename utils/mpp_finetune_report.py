"""
Shared helpers for MPP finetune search reporting (wide candidate table + stage long table).
Used by 3_finetune_2step_mpp.py, 3_finetune_3step_mpp.py, and 3.1_finetune_search_aggregate.py.
"""
import json
from typing import Any, Dict, List, Optional


def flatten_metric(prefix: str, metric: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if not metric:
        return row
    primary_metric = metric.get("primary_metric")
    row[f"{prefix}_task_type"] = metric.get("task_type")
    row[f"{prefix}_primary_metric"] = primary_metric
    row[f"{prefix}_primary_score"] = metric.get(primary_metric) if primary_metric else None
    for key, value in metric.items():
        if key not in {"task_type", "primary_metric"}:
            row[f"{prefix}_{key}"] = value
    return row


def build_candidate_row(candidate: Dict[str, Any]) -> Dict[str, Any]:
    params = candidate["params"]
    row: Dict[str, Any] = {
        "idx": candidate["idx"],
        "is_best": candidate.get("is_best", False),
        "is_best_among_scanned": candidate.get("is_best_among_scanned", candidate.get("is_best", False)),
        "trial_dir": candidate.get("trial_dir"),
        "completion": candidate.get("completion"),
        "single_stage": params.get("single_stage", False),
        "run_tag": candidate["run_tag"],
        "stage1_run_tag": candidate.get("stage1_run_tag"),
        "stage2_run_tag": candidate.get("stage2_run_tag"),
        "stage3_run_tag": candidate.get("stage3_run_tag"),
        "stage1_model": candidate.get("stage1_model"),
        "stage2_model": candidate.get("stage2_model"),
        "stage3_model": candidate.get("stage3_model"),
        "selected_model": candidate.get("model"),
        "final_weights_for_next_step": candidate.get("final_weights"),
        "seed": params.get("seed"),
        "batch_size": params.get("batch_size"),
        "dropout": params.get("dropout"),
        "weight_decay": params.get("weight_decay"),
        "init_base_lr": params.get("init_base_lr"),
        "optim_type": params.get("optim_type"),
        "warm_up_epoch": params.get("warm_up_epoch"),
        "start_lr": params.get("start_lr"),
        "scheduler_type": params.get("scheduler_type"),
        "stage1_lr": params.get("stage1_lr"),
        "stage2_lr": params.get("stage2_lr"),
        "stage3_lr": params.get("stage3_lr"),
        "stage1_epochs": params.get("stage1_epochs"),
        "stage2_epochs": params.get("stage2_epochs"),
        "stage3_epochs": params.get("stage3_epochs"),
        "stage1_patience": params.get("stage1_patience"),
        "stage2_patience": params.get("stage2_patience"),
        "stage3_patience": params.get("stage3_patience"),
        "dist_bar": json.dumps(params.get("dist_bar"), ensure_ascii=False),
        "loss_weights": json.dumps(params.get("loss_weights"), ensure_ascii=False),
        "regression_label_transform": params.get("regression_label_transform"),
    }
    row.update(flatten_metric("stage1_valid", candidate.get("stage1_valid")))
    row.update(flatten_metric("stage1_test", candidate.get("stage1_test")))
    if candidate.get("stage3_model"):
        row.update(flatten_metric("stage2_valid", candidate.get("stage2_valid")))
        row.update(flatten_metric("stage2_test", candidate.get("stage2_test")))
        row.update(flatten_metric("stage3_valid", candidate.get("valid")))
        row.update(flatten_metric("stage3_test", candidate.get("test")))
    else:
        row.update(flatten_metric("stage2_valid", candidate.get("valid")))
        row.update(flatten_metric("stage2_test", candidate.get("test")))
    return row


def _metric_split_columns(split: str, metric: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat valid/test columns for stage score table (regression + classification)."""
    out: Dict[str, Any] = {}
    if not metric:
        return out
    tt = metric.get("task_type")
    if tt == "regression":
        for k in ("mae", "rmse", "r2"):
            if k in metric:
                out[f"{split}_{k}"] = metric[k]
    elif tt == "classification":
        if "roc_auc" in metric:
            out[f"{split}_roc_auc"] = metric["roc_auc"]
        if "pr_auc" in metric:
            out[f"{split}_pr_auc"] = metric["pr_auc"]
    return out


def build_stage_score_rows(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = candidate["params"]
    base = {
        "idx": candidate["idx"],
        "single_stage": params.get("single_stage", False),
        "completion": candidate.get("completion"),
        "seed": params.get("seed"),
        "batch_size": params.get("batch_size"),
        "dropout": params.get("dropout"),
        "weight_decay": params.get("weight_decay"),
        "scheduler_type": params.get("scheduler_type"),
        "dist_bar": json.dumps(params.get("dist_bar"), ensure_ascii=False),
        "regression_label_transform": params.get("regression_label_transform"),
    }
    stage_specs = []
    if params.get("single_stage", False):
        stage_specs.append(("single", candidate.get("model"), candidate.get("valid"), candidate.get("test")))
    elif candidate.get("stage3_model"):
        stage_specs.extend([
            ("stage1", candidate.get("stage1_model"), candidate.get("stage1_valid"), candidate.get("stage1_test")),
            ("stage2", candidate.get("stage2_model"), candidate.get("stage2_valid"), candidate.get("stage2_test")),
            ("stage3", candidate.get("stage3_model"), candidate.get("valid"), candidate.get("test")),
        ])
    else:
        stage_specs.extend([
            ("stage1", candidate.get("stage1_model"), candidate.get("stage1_valid"), candidate.get("stage1_test")),
            ("stage2", candidate.get("stage2_model"), candidate.get("valid"), candidate.get("test")),
        ])

    rows: List[Dict[str, Any]] = []
    for stage_name, model_path, valid_metric, test_metric in stage_specs:
        row = dict(base)
        row.update({
            "stage": stage_name,
            "model_path": model_path,
            "is_selected_stage": model_path == candidate.get("model"),
            "is_best": candidate.get("is_best", False) and model_path == candidate.get("model"),
            "lr": params.get(f"{stage_name}_lr") if stage_name in ["stage1", "stage2", "stage3"] else params.get("stage3_lr", params.get("stage2_lr")),
            "epochs": params.get(f"{stage_name}_epochs") if stage_name in ["stage1", "stage2", "stage3"] else params.get("stage3_epochs", params.get("stage2_epochs")),
            "patience": params.get(f"{stage_name}_patience") if stage_name in ["stage1", "stage2", "stage3"] else params.get("stage3_patience", params.get("stage2_patience")),
        })
        row.update(_metric_split_columns("valid", valid_metric))
        row.update(_metric_split_columns("test", test_metric))
        rows.append(row)
    return rows


def primary_valid_score_for_ranking(valid_metric: Optional[Dict[str, Any]]) -> Optional[float]:
    """Match 3_finetune_2step_mpp: classification -> roc_auc, regression -> r2."""
    if not valid_metric or valid_metric.get("task_type") == "unknown":
        return None
    tt = valid_metric.get("task_type")
    if tt == "classification":
        return valid_metric.get("roc_auc")
    if tt == "regression":
        return valid_metric.get("r2")
    return None
