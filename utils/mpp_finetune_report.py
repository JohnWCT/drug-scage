"""
Shared helpers for MPP finetune search reporting (wide candidate table + stage long table).
Used by 3_finetune_2step_mpp.py, 3_finetune_3step_mpp.py, and 3.1_finetune_search_aggregate.py.
"""
import json
from typing import Any, Dict, List, Optional


def _load_checkpoint_metadata(model_path: Optional[str]) -> Dict[str, Any]:
    if not model_path:
        return {}
    try:
        import torch

        ckpt = torch.load(model_path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(ckpt, dict):
        return {}
    metadata = ckpt.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _metadata_metric_columns(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}

    row = {
        "selection_metric": metadata.get("selection_metric"),
        "best_epoch": metadata.get("best_epoch"),
        "best_valid_metric": metadata.get("best_valid_metric"),
        "has_test_set": metadata.get("has_test_set"),
        "metadata_split_pkl_path": metadata.get("split_pkl_path"),
        "metadata_encoder_lr_ratio": metadata.get("encoder_lr_ratio"),
        "metadata_max_grad_norm": metadata.get("max_grad_norm"),
        "metadata_mt_loss_tau": metadata.get("mt_loss_tau"),
        "metadata_mt_ratio_clip": metadata.get("mt_ratio_clip"),
        "metadata_mt_loss_prior": json.dumps(metadata.get("mt_loss_prior"), ensure_ascii=False),
    }
    if metadata.get("selection_metric") == "valid_auc":
        row.update({
            "valid_roc_auc": metadata.get("best_valid_auc"),
            "test_roc_auc": metadata.get("best_test_auc"),
        })
    else:
        row.update({
            "valid_mae": metadata.get("best_valid_mae"),
            "valid_rmse": metadata.get("best_valid_rmse"),
            "valid_r2": metadata.get("best_valid_r2"),
            "test_mae": metadata.get("best_test_mae"),
            "test_rmse": metadata.get("best_test_rmse"),
            "test_r2": metadata.get("best_test_r2"),
        })
    return row


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


def _json_or_none(value: Any) -> Optional[str]:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def _hidden_param_columns(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "split_pkl_path": params.get("split_pkl_path"),
        "allow_empty_test": params.get("allow_empty_test"),
        "encoder_lr_ratio": params.get("encoder_lr_ratio"),
        "max_grad_norm": params.get("max_grad_norm"),
        "mt_loss_eps": params.get("mt_loss_eps"),
        "mt_loss_tau": params.get("mt_loss_tau"),
        "mt_ratio_clip": params.get("mt_ratio_clip"),
        "mt_loss_prior": _json_or_none(params.get("mt_loss_prior")),
        "winsorize_quantile_low": params.get("winsorize_quantile_low"),
        "winsorize_quantile_high": params.get("winsorize_quantile_high"),
        "graph_label_reweight": params.get("graph_label_reweight"),
        "graph_label_bins": _json_or_none(params.get("graph_label_bins")),
        "graph_label_weight_power": params.get("graph_label_weight_power"),
        "graph_label_weight_clip": params.get("graph_label_weight_clip"),
        "warm_up_epoch": params.get("warm_up_epoch"),
        "init_base_lr": params.get("init_base_lr"),
        "optim_type": params.get("optim_type"),
        "start_lr": params.get("start_lr"),
        "stage2_warmup_epochs": params.get("stage2_warmup_epochs"),
        "stage3_warmup_epochs": params.get("stage3_warmup_epochs"),
        "stage1_loss_weights": _json_or_none(params.get("stage1_loss_weights")),
        "stage2_loss_weights": _json_or_none(params.get("stage2_loss_weights")),
        "stage3_loss_weights": _json_or_none(params.get("stage3_loss_weights")),
    }


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
    row.update(_hidden_param_columns(params))
    row["has_test_set"] = not bool(params.get("allow_empty_test", False))
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
        "trial_serial": params.get("trial_serial", candidate.get("idx")),
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
    base.update(_hidden_param_columns(params))
    base["has_test_set"] = not bool(params.get("allow_empty_test", False))
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
        metadata = _load_checkpoint_metadata(model_path)
        row.update(_metadata_metric_columns(metadata))
        rows.append(row)
    return rows


def primary_valid_score_for_ranking(valid_metric: Optional[Dict[str, Any]]) -> Optional[float]:
    """Primary ranking metric: classification -> roc_auc, regression -> rmse."""
    if not valid_metric or valid_metric.get("task_type") == "unknown":
        return None
    tt = valid_metric.get("task_type")
    if tt == "classification":
        return valid_metric.get("roc_auc")
    if tt == "regression":
        return valid_metric.get("rmse")
    return None
