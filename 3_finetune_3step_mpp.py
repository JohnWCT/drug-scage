"""
Three-step MPP finetuning runner for SCAGE.

流程:
  1. Stage 1: freeze encoder，只訓練 head_Graph。
  2. Stage 2: 載入 Stage 1 best model，freeze encoder，訓練 head_Graph、finger head、atom FG head。
  3. Stage 3: 載入 Stage 2 best model，解凍全模型 finetune。
  4. 依 valid split 的 primary metric 選出最佳參數組合，複製到 result_dir/<task>_best.pth。

執行範例:
  python 3_finetune_3step_mpp.py --search_config ./config/caco2_finetune_search.yaml

  python 3_finetune_3step_mpp.py \
    --task caco2 \
    --dataroot ./data/mpp/pkl \
    --splitroot ./data/mpp/split \
    --split_type random_scaffold \
    --result_dir ./outputs/caco2_pipeline \
    --gpus 0 \
    --search_config ./config/caco2_finetune_search.yaml

  python 3_finetune_3step_mpp.py \
    --task caco2 \
    --dataroot ./data/mpp/pkl \
    --splitroot ./data/mpp/split \
    --split_type random_scaffold \
    --single_stage

參數搜尋:
  --search_config 支援 JSON 或 YAML。檔案內的 search 區塊可以放 list，
  程式會用 Cartesian product 暴力搜尋所有組合。
"""

import argparse
import gc
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import finetune_mpp as upstream
from _config import *
from data_process.data_collator import collator_finetune_pkl
from data_process.split import create_splitter
from datasets.dataloader import FinetuneDataset as FinetuneDataset_pkl
from utils.caco2_pipeline_utils import (
    copy_model_to_result_dir,
    resolve_path,
    run_predict_eval_metrics_only,
    save_scores,
    settings_namespace,
)
from utils.mpp_finetune_report import build_stage_score_rows
from utils.global_var_util import *
from utils.loss_util import get_balanced_atom_fg_loss
from utils.metric_util import compute_cls_metric_tensor, compute_reg_metric_with_r2
from utils.public_util import EarlyStopping, set_seed
from utils.userconfig_util import config_current_user, config_dataset_form, drop_last_flag, get_dataset_form


def parse_args():
    parser = argparse.ArgumentParser(description="Three-stage/custom MPP finetune runner")
    parser.add_argument("--search_config", type=str, default=None,
                        help="JSON/YAML file containing fixed params and search lists")
    parser.add_argument("--task", type=str, default="caco2")
    parser.add_argument("--dataroot", type=str, default="./data/mpp/pkl")
    parser.add_argument("--splitroot", type=str, default="./data/mpp/split")
    parser.add_argument(
        "--split_type",
        type=str,
        default="random_scaffold",
        choices=["scaffold", "random_scaffold", "cyclic_scaffold"],
    )
    parser.add_argument("--split_pkl_path", type=str, default=None)
    parser.add_argument("--allow_empty_test", action="store_true")
    parser.add_argument("--result_dir", type=str, default="./outputs/caco2_pipeline")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--single_stage", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default="./weights/pretrain/pretrain.pth")
    parser.add_argument("--save_ckpt", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--optim_type", type=str, default="adam", choices=["adam", "rms", "sgd"])
    parser.add_argument("--scheduler_type", type=str, default="None", choices=["None", "linear", "square", "cos"])
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--dist_bar", nargs="+", type=int, default=[30, 60],
                        help="Default multiscale distance percentile thresholds")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layer_num", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--minimal_outputs", type=int, default=1, choices=[0, 1],
                        help="1: keep model.pth + config_finetune.yaml + learning_curve.png + record.txt")
    parser.add_argument(
        "--resume_disk",
        action="store_true",
        help="If trial_dir already has trained model.pth, skip training and only re-run predict eval (for crash recovery).",
    )
    return parser.parse_args()


def custom_log_dir(config: Dict) -> str:
    if config.get("log_dir"):
        return str(config["log_dir"])
    run_name = (
        f"{config['task_name']}_{config['seed']}_{config['split_type']}_"
        f"{config['optim']['init_lr']}_{config['batch_size']}"
    )
    if config.get("run_tag"):
        run_name = f"{run_name}_{config['run_tag']}"
    return os.path.join("finetune_result", run_name)


def read_search_config(path: str):
    if path is None:
        return {}
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        if cfg_path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def as_list(value, default):
    if value is None:
        value = default
    return value if isinstance(value, list) else [value]


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(data), f, sort_keys=False, allow_unicode=True)


def config_name_tag(search_config: str) -> str:
    if not search_config:
        return "default_config"
    stem = Path(search_config).stem.strip()
    if not stem:
        return "default_config"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe[:80] if safe else "default_config"


def trial_folder_name(task: str, split_type: str, config_tag: str, idx: int) -> str:
    return f"{task}_{split_type}_{config_tag}_{idx:03d}"


def save_trial_params(trial_dir: Path, args, trial_cfg, stage_trials=None):
    payload = {
        "task": args.task,
        "split_type": args.split_type,
        "trial_serial": trial_cfg.get("trial_serial"),
        "trial_name": trial_cfg.get("trial_name"),
        "trial_dir": str(trial_dir),
        "search_trial": trial_cfg,
    }
    if stage_trials:
        payload["stages"] = stage_trials
    write_yaml(trial_dir / "trial_params.yaml", payload)
    if stage_trials:
        for stage_name, stage_trial in stage_trials.items():
            write_yaml(trial_dir / f"{stage_name}_params.yaml", stage_trial)


def compact_metric_dict(metric_dict):
    if "rmse" in metric_dict:
        return {
            "mae": f"{metric_dict['mae']:.4f}",
            "rmse": f"{metric_dict['rmse']:.4f}",
            "r2": f"{metric_dict['r2']:.4f}",
        }
    if "auc" in metric_dict:
        return {"auc": f"{metric_dict['auc']:.4f}"}
    return {key: f"{value:.4f}" for key, value in metric_dict.items()}


class CustomTrainer(upstream.Trainer):
    def __init__(self, config, file_path):
        super().__init__(config, file_path)
        self.regression_target_transform = self._build_regression_target_transform()
        self.last_mt_weights = None
        self.best_metadata = {}
        self.graph_label_reweight_info = self._build_graph_label_reweight_info()

    def get_data_loaders(self):
        dataset = FinetuneDataset_pkl(root=self.config["root"], task_name=self.config["task_name"])
        splitter = create_splitter(
            self.config["split_type"],
            self.config["seed"],
            split_pkl_path=self.config.get("split_pkl_path"),
            allow_empty_test=self.config.get("allow_empty_test", False),
        )
        train_dataset, val_dataset, test_dataset = splitter.split(dataset, self.config["task_name"])

        num_workers = self.config["dataloader_num_workers"]
        pin_memory = bool(self.config.get("pin_memory", False))
        if num_workers == 0:
            pin_memory = False
        bsz = self.config["batch_size"]
        train_loader = DataLoader(
            train_dataset,
            batch_size=bsz,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator_finetune_pkl,
            pin_memory=pin_memory,
            drop_last=drop_last_flag(len(train_dataset), bsz),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator_finetune_pkl,
            pin_memory=pin_memory,
            drop_last=drop_last_flag(len(val_dataset), bsz),
        )
        if test_dataset is None:
            test_loader = None
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=bsz,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collator_finetune_pkl,
                drop_last=drop_last_flag(len(test_dataset), bsz),
            )
        self.has_test_set = test_loader is not None and len(test_loader.dataset) > 0
        return train_loader, val_loader, test_loader

    def calc_dynamic_mt_loss(self, loss_list, update_state=True):
        loss_list = torch.stack(loss_list)
        eps = float(self.config.get("mt_loss_eps", 1e-8))
        tau = float(self.config.get("mt_loss_tau", 2.0))
        ratio_clip = float(self.config.get("mt_ratio_clip", 5.0))
        step = int(self.cur_loss_step.item())

        if step == 0 and not update_state:
            return loss_list.mean()

        if step == 0:
            denom = loss_list.detach().clamp_min(eps)
            if update_state:
                self.loss_init[:, 0] = loss_list.detach()
                self.loss_last2[:, 0] = loss_list.detach()
                self.cur_loss_step += 1
            self.last_mt_weights = torch.ones_like(loss_list).detach() / loss_list.numel()
            return (loss_list / denom).mean()

        if step == 1:
            denom = self.loss_init[:, :1].mean(dim=-1).clamp_min(eps)
            if update_state:
                self.loss_last[:, 0] = loss_list.detach()
                self.loss_init[:, 1] = loss_list.detach()
                self.cur_loss_step += 1
            self.last_mt_weights = torch.ones_like(loss_list).detach() / loss_list.numel()
            return (loss_list / denom).mean()

        init_cols = min(step, self.batch_considered)
        last_cols = min(max(step - 1, 1), self.batch_considered // 10)
        cur_loss_init = self.loss_init[:, :init_cols].mean(dim=-1).clamp_min(eps)
        cur_loss_last = self.loss_last[:, :last_cols].mean(dim=-1).clamp_min(eps)
        cur_loss_last2 = self.loss_last2[:, :last_cols].mean(dim=-1).clamp_min(eps)
        ratio = (cur_loss_last / cur_loss_last2).clamp(min=1.0 / ratio_clip, max=ratio_clip)
        weights = F.softmax(ratio / tau, dim=-1).detach()
        prior = self.config.get("mt_loss_prior")
        if prior is not None:
            prior_tensor = torch.tensor(
                [
                    float(prior.get("graph", 1.0)),
                    float(prior.get("finger", 1.0)),
                    float(prior.get("atom_fg", 1.0)),
                ],
                dtype=weights.dtype,
                device=weights.device,
            )
            weights = weights * prior_tensor
            weights = weights / weights.sum().clamp_min(eps)
        self.last_mt_weights = weights.detach()
        total_loss = (loss_list / cur_loss_init * weights).sum()

        if update_state:
            cur_init_idx = step % self.batch_considered
            self.loss_init[:, cur_init_idx] = loss_list.detach()
            cur_last_idx = (step - 1) % (self.batch_considered // 10)
            prev_last_idx = (cur_last_idx - 1) % (self.batch_considered // 10)
            self.loss_last2[:, cur_last_idx] = self.loss_last[:, prev_last_idx]
            self.loss_last[:, cur_last_idx] = loss_list.detach()
            self.cur_loss_step += 1
        return total_loss

    def calc_mt_loss(self, loss_list, update_state=True):
        loss_weights = self.config.get("loss_weights")
        if loss_weights:
            loss_list = torch.stack(loss_list)
            weights = torch.tensor(
                [
                    float(loss_weights.get("graph", 1.0)),
                    float(loss_weights.get("finger", 1.0)),
                    float(loss_weights.get("atom_fg", 1.0)),
                ],
                dtype=loss_list.dtype,
                device=loss_list.device,
            )
            active = torch.sum(weights)
            if active <= 0:
                raise ValueError("At least one loss weight must be > 0.")
            self.last_mt_weights = (weights / active).detach()
            return torch.sum(loss_list * weights) / active
        return self.calc_dynamic_mt_loss(loss_list, update_state=update_state)

    def _build_regression_target_transform(self):
        transform_type = self.config.get("regression_label_transform", "none")
        if self.config["task"] != "regression" or transform_type in [None, "none", "None"]:
            return {"type": "none"}
        if transform_type not in {"standardize", "winsorized_standardize"}:
            raise ValueError(f"Unsupported regression_label_transform: {transform_type}")

        labels = []
        for batch in self.train_loader:
            label = batch["label"].float().view(-1, self.config["num_tasks"])
            label = label[torch.isfinite(label).all(dim=1)]
            if label.numel() > 0:
                labels.append(label)
        if not labels:
            raise ValueError("Cannot compute regression label statistics from an empty training split.")

        label_tensor = torch.cat(labels, dim=0)
        if transform_type == "winsorized_standardize":
            q_low = float(self.config.get("winsorize_quantile_low", 0.01))
            q_high = float(self.config.get("winsorize_quantile_high", 0.99))
            low = torch.quantile(label_tensor, q_low, dim=0)
            high = torch.quantile(label_tensor, q_high, dim=0)
            clipped = torch.clamp(label_tensor, min=low, max=high)
            mean = clipped.mean(dim=0)
            std = clipped.std(dim=0, unbiased=False).clamp_min(1e-6)
            transform = {
                "type": "winsorized_standardize",
                "mean": mean.tolist(),
                "std": std.tolist(),
                "low": low.tolist(),
                "high": high.tolist(),
                "q_low": q_low,
                "q_high": q_high,
            }
        else:
            mean = label_tensor.mean(dim=0)
            std = label_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
            transform = {
                "type": "standardize",
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        print(f"[INFO] regression_label_transform={transform}")
        write_record(self.txtfile, f"regression_label_transform:{transform}")
        return transform

    def _target_stats_tensor(self, key: str, like: Tensor) -> Tensor:
        values = self.regression_target_transform[key]
        return torch.tensor(values, dtype=like.dtype, device=like.device).view(1, -1)

    def _transform_regression_label(self, label: Tensor) -> Tensor:
        label = label.view(-1, self.config["num_tasks"])
        transform_type = self.regression_target_transform.get("type")
        if transform_type == "none":
            return label
        if transform_type == "winsorized_standardize":
            low = self._target_stats_tensor("low", label)
            high = self._target_stats_tensor("high", label)
            label = torch.clamp(label, min=low, max=high)
        mean = self._target_stats_tensor("mean", label)
        std = self._target_stats_tensor("std", label)
        return (label - mean) / std

    def _inverse_regression_pred(self, pred: Tensor) -> Tensor:
        transform_type = self.regression_target_transform.get("type")
        if transform_type not in {"standardize", "winsorized_standardize"}:
            return pred
        mean = self._target_stats_tensor("mean", pred)
        std = self._target_stats_tensor("std", pred)
        return pred * std + mean

    def _build_graph_label_reweight_info(self):
        if self.config["task"] != "regression" or not self.config.get("graph_label_reweight", False):
            return None

        bins = torch.tensor(
            self.config.get("graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]),
            dtype=torch.float32,
        )
        if bins.numel() < 2:
            return None

        labels = []
        for batch in self.train_loader:
            label = batch["label"].float().view(-1)
            label = label[torch.isfinite(label)]
            if label.numel() > 0:
                labels.append(label.cpu())
        if not labels:
            return None

        labels = torch.cat(labels, dim=0)
        bin_idx = torch.bucketize(labels, bins) - 1
        bin_idx = bin_idx.clamp(min=0, max=len(bins) - 2)
        counts = torch.bincount(bin_idx, minlength=len(bins) - 1).float().clamp_min(1.0)

        power = float(self.config.get("graph_label_weight_power", 0.5))
        clip_value = float(self.config.get("graph_label_weight_clip", 3.0))
        weights = counts.pow(-power)
        weights = weights / weights.mean().clamp_min(1e-8)
        weights = weights.clamp(max=clip_value)

        info = {"bins": bins.tolist(), "weights": weights.tolist()}
        write_record(self.txtfile, f"graph_label_reweight_info:{info}")
        return info

    def _graph_label_weights(self, raw_label: Tensor):
        info = self.graph_label_reweight_info
        if info is None:
            return None
        bins = torch.tensor(info["bins"], dtype=raw_label.dtype, device=raw_label.device)
        weights = torch.tensor(info["weights"], dtype=raw_label.dtype, device=raw_label.device)
        y = raw_label.float().view(-1)
        bin_idx = torch.bucketize(y, bins) - 1
        bin_idx = bin_idx.clamp(min=0, max=weights.numel() - 1)
        return weights[bin_idx].view_as(raw_label)

    def _checkpoint_payload(self):
        payload = {
            "model": self.net.state_dict(),
            "metadata": getattr(self, "best_metadata", {}),
        }
        if self.config["task"] == "regression":
            payload["target_transform"] = self.regression_target_transform
        return payload

    @staticmethod
    def _metric_float(metric_dict: Dict, key: str):
        value = metric_dict.get(key)
        return float(value) if value is not None else float(np.nan)

    def _current_dist_bar(self):
        if GlobalVar.dist_bar is None:
            return None
        return list(np.asarray(GlobalVar.dist_bar).tolist())

    def _mt_weights_dict(self):
        if self.last_mt_weights is None:
            return None
        weights = self.last_mt_weights.detach().cpu()
        return {
            "graph": float(weights[0]),
            "finger": float(weights[1]),
            "atom_fg": float(weights[2]),
        }

    def _update_best_metadata(
        self,
        epoch,
        valid_select_metric,
        train_metrics,
        valid_metrics,
        test_metrics,
    ):
        if self.config["task"] == "classification":
            self.best_metadata = {
                "selection_metric": "valid_auc",
                "best_epoch": int(epoch),
                "best_valid_metric": float(valid_select_metric),
                "best_valid_auc": self._metric_float(valid_metrics, "auc"),
                "best_test_auc": self._metric_float(test_metrics, "auc"),
                "has_test_set": bool(getattr(self, "has_test_set", True)),
                "split_pkl_path": self.config.get("split_pkl_path"),
                "stage_name": self.config.get("stage_name"),
                "trial_serial": self.config.get("trial_serial"),
                "dist_bar": self._current_dist_bar(),
                "loss_weights": self.config.get("loss_weights"),
                "mt_loss_tau": self.config.get("mt_loss_tau"),
                "mt_ratio_clip": self.config.get("mt_ratio_clip"),
                "mt_loss_prior": self.config.get("mt_loss_prior"),
                "encoder_lr_ratio": self.config.get("encoder_lr_ratio"),
                "max_grad_norm": self.config.get("max_grad_norm"),
            }
            return

        self.best_metadata = {
            "selection_metric": "valid_rmse",
            "best_epoch": int(epoch),
            "best_valid_metric": float(valid_select_metric),
            "best_train_rmse": self._metric_float(train_metrics, "rmse"),
            "best_train_mae": self._metric_float(train_metrics, "mae"),
            "best_train_r2": self._metric_float(train_metrics, "r2"),
            "best_valid_rmse": self._metric_float(valid_metrics, "rmse"),
            "best_valid_mae": self._metric_float(valid_metrics, "mae"),
            "best_valid_r2": self._metric_float(valid_metrics, "r2"),
            "best_test_rmse": self._metric_float(test_metrics, "rmse"),
            "best_test_mae": self._metric_float(test_metrics, "mae"),
            "best_test_r2": self._metric_float(test_metrics, "r2"),
            "has_test_set": bool(getattr(self, "has_test_set", True)),
            "split_pkl_path": self.config.get("split_pkl_path"),
            "stage_name": self.config.get("stage_name"),
            "trial_serial": self.config.get("trial_serial"),
            "dist_bar": self._current_dist_bar(),
            "loss_weights": self.config.get("loss_weights"),
            "mt_loss_tau": self.config.get("mt_loss_tau"),
            "mt_ratio_clip": self.config.get("mt_ratio_clip"),
            "mt_loss_prior": self.config.get("mt_loss_prior"),
            "encoder_lr_ratio": self.config.get("encoder_lr_ratio"),
            "max_grad_norm": self.config.get("max_grad_norm"),
        }

    @staticmethod
    def _is_trainable_head(name: str) -> bool:
        return (
            "head_Graph" in name
            or "head_finger_keeping_atom" in name
            or "head_FG_atom" in name
        )

    def _get_optim(self):
        trainable_scope = self.config.get("trainable_scope", "all")
        frozen_names = []
        for name, param in self.net.named_parameters():
            if trainable_scope == "head":
                param.requires_grad = "head_Graph" in name
            elif trainable_scope == "heads":
                param.requires_grad = self._is_trainable_head(name)
            elif trainable_scope == "all":
                param.requires_grad = True
            else:
                raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")
            if not param.requires_grad:
                frozen_names.append(name)

        optim_type = self.config["optim"]["type"]
        lr = self.config["optim"]["init_lr"]
        weight_decay = self.config["optim"]["weight_decay"]
        encoder_lr_ratio = float(self.config.get("encoder_lr_ratio", 1.0))

        # Differential LR: encoder uses lr * encoder_lr_ratio, head uses lr
        if trainable_scope == "all" and encoder_lr_ratio < 1.0:
            encoder_params = []
            head_params = []
            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    continue
                if self._is_trainable_head(name):
                    head_params.append(param)
                else:
                    encoder_params.append(param)
            encoder_lr = lr * encoder_lr_ratio
            param_groups = [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": head_params, "lr": lr},
            ]
            total_trainable = len(encoder_params) + len(head_params)
            if total_trainable == 0:
                raise ValueError("No trainable parameters selected.")
            print(
                f"[INFO] trainable_scope={trainable_scope}, frozen_params={len(frozen_names)}, "
                f"encoder_params={len(encoder_params)} (lr={encoder_lr:.2e}), "
                f"head_params={len(head_params)} (lr={lr:.2e}), "
                f"encoder_lr_ratio={encoder_lr_ratio}"
            )
        else:
            params = [param for _, param in self.net.named_parameters() if param.requires_grad]
            if not params:
                raise ValueError("No trainable parameters selected.")
            param_groups = [{"params": params}]
            print(f"[INFO] trainable_scope={trainable_scope}, frozen_params={len(frozen_names)}")

        if optim_type == "adam":
            return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        if optim_type == "rms":
            return torch.optim.RMSprop(param_groups, lr=lr, weight_decay=weight_decay)
        if optim_type == "sgd":
            momentum = self.config["optim"].get("momentum", 0)
            return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
        raise ValueError("not supported optimizer!")

    def _step(self, model, batch: Dict[str, Tensor], update_mt_state=True):
        dataset_form: str = get_dataset_form()
        pred_dict: Dict = model(batch)
        pred = pred_dict["graph_feature"]
        finger = pred_dict["finger_feature"]
        atom_fg = pred_dict["atom_fg"]

        loss_weights = self.config.get("loss_weights") or {}
        finger_weight = float(loss_weights.get("finger", 1.0))
        atom_fg_weight = float(loss_weights.get("atom_fg", 1.0))
        zero_loss = pred.sum() * 0.0

        if atom_fg_weight > 0:
            loss_atom_fg = get_balanced_atom_fg_loss(
                atom_fg,
                batch["function_group_index"],
                loss_f_atom_fg=F.binary_cross_entropy_with_logits,
            )
        else:
            loss_atom_fg = zero_loss
        if finger_weight > 0:
            loss_finger = F.binary_cross_entropy_with_logits(finger, batch["morgan2048_fp"].float())
        else:
            loss_finger = zero_loss

        if self.config["task"] == "classification":
            label: Tensor = batch["label"]
            if dataset_form == "pyg":
                is_valid: Tensor = label ** 2 > 0
                label = ((label + 1.0) / 2).view(pred.shape)
            elif dataset_form == "pkl":
                is_valid: Tensor = label >= 0
                label = (label + 0.0).view(pred.shape)
            else:
                raise ValueError("not supported dataset form!")
            loss_graph = self.criterion(pred, label)
            loss_graph = torch.where(
                is_valid,
                loss_graph,
                torch.zeros(loss_graph.shape, device="cuda").to(loss_graph.dtype),
            )
            loss_graph = torch.sum(loss_graph) / torch.sum(is_valid)
        else:
            raw_label = batch["label"].float().view(-1, self.config["num_tasks"])
            label = self._transform_regression_label(raw_label)
            pred_view = pred.view(label.shape)
            loss_graph_element = F.mse_loss(pred_view, label, reduction="none")
            sample_weights = self._graph_label_weights(raw_label)
            if sample_weights is not None:
                sample_weights = sample_weights.to(loss_graph_element.device).view_as(loss_graph_element)
                loss_graph = (loss_graph_element * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            else:
                loss_graph = loss_graph_element.mean()

        total_loss = self.calc_mt_loss(
            [loss_graph, loss_finger, loss_atom_fg],
            update_state=update_mt_state,
        )
        raw_total_loss = loss_graph + loss_finger + loss_atom_fg
        raw_mean_loss = raw_total_loss / 3.0
        loss_parts = {
            "graph_loss": float(loss_graph.detach().item()),
            "finger_loss": float(loss_finger.detach().item()),
            "atom_fg_loss": float(loss_atom_fg.detach().item()),
            "raw_total_loss": float(raw_total_loss.detach().item()),
            "raw_mean_loss": float(raw_mean_loss.detach().item()),
        }
        return total_loss, pred, loss_parts

    def _compute_metrics(self, y_true: Tensor, y_pred: Tensor):
        if self.config["task"] == "regression":
            y_pred = self._inverse_regression_pred(y_pred)
            mae, rmse, r2 = compute_reg_metric_with_r2(y_true, y_pred)
            return rmse, {"mae": mae, "rmse": rmse, "r2": r2}
        roc_auc = compute_cls_metric_tensor(y_true, y_pred)
        return roc_auc, {"auc": roc_auc}

    def _train_step(self):
        self.net.train()
        num_data = 0
        train_loss = 0
        loss_parts_sum = {key: 0.0 for key in self._loss_part_keys()}
        y_pred: Tensor = Tensor().to("cuda")
        y_true: Tensor = Tensor().to("cuda")
        progress = tqdm(
            self.train_loader,
            desc=self._progress_label("train"),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(progress, start=1):
            self.optim.zero_grad()
            batch = {key: value.to("cuda") for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch["edge_weight"] = None

            loss, pred, loss_parts = self._step(self.net, batch, update_mt_state=True)
            train_loss += loss.item()
            for key in loss_parts_sum:
                loss_parts_sum[key] += loss_parts[key]
            self.writer.add_scalar("train_loss", loss, global_step=self.optim_steps)
            y_pred = torch.cat([y_pred, pred])
            y_true = torch.cat([y_true, batch["label"]])
            loss.backward()
            max_grad_norm = self.config.get("max_grad_norm", 0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.net.parameters() if p.requires_grad],
                    max_norm=max_grad_norm,
                )
            self.optim.step()
            num_data += 1
            self.optim_steps += 1
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(self.train_loader):
                progress.set_postfix(self._batch_postfix("train", train_loss, loss_parts_sum, num_data))

        train_loss /= num_data
        torch.cuda.empty_cache()
        metric, metric_dict = self._compute_metrics(y_true, y_pred)
        loss_parts_avg = {key: value / num_data for key, value in loss_parts_sum.items()}
        return train_loss, metric, loss_parts_avg, metric_dict

    def _valid_step(self, valid_loader, split_name="valid"):
        self.net.eval()
        y_pred = Tensor().to("cuda")
        y_true = Tensor().to("cuda")
        valid_loss = 0
        num_data = 0
        loss_parts_sum = {key: 0.0 for key in self._loss_part_keys()}
        progress = tqdm(
            valid_loader,
            desc=self._progress_label(split_name),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(progress, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch["edge_weight"] = None
            with torch.no_grad():
                loss, pred, loss_parts = self._step(self.net, batch, update_mt_state=False)
            valid_loss += loss.item()
            for key in loss_parts_sum:
                loss_parts_sum[key] += loss_parts[key]
            num_data += 1
            y_pred = torch.cat([y_pred, pred])
            y_true = torch.cat([y_true, batch["label"]])
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(valid_loader):
                progress.set_postfix(self._batch_postfix(split_name, valid_loss, loss_parts_sum, num_data))

        valid_loss /= num_data
        metric, metric_dict = self._compute_metrics(y_true, y_pred)
        loss_parts_avg = {key: value / num_data for key, value in loss_parts_sum.items()}
        return valid_loss, metric, loss_parts_avg, metric_dict

    def load_ckpt(self, load_pth):
        if GlobalVar.use_ckpt:
            print(f"load model from {load_pth}")
            loaded = torch.load(load_pth, map_location="cuda")
            checkpoint = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
            new_model_dict = self.net.state_dict()
            new_model_keys = set(list(new_model_dict.keys()))
            pretrained_dict = {".".join(k.split(".")): v for k, v in checkpoint.items()}
            pretrained_keys = set(list(".".join(k.split(".")) for k in checkpoint.keys()))
            if model_is_dp(pretrained_keys):
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                pretrained_keys = set(list(k[7:] for k in pretrained_keys))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_keys}
            new_model_dict.update(pretrained_dict)
            self.net.load_state_dict(new_model_dict)
            if "dist_bar" in checkpoint:
                GlobalVar.dist_bar = checkpoint["dist_bar"].cpu().numpy()

        self.start_epoch = 1
        self.optim_steps = 0
        self.best_metric = self._initial_best_metric()
        self.writer = upstream.create_fresh_writer(self.config)

    def _is_better(self, metric, best_metric):
        if metric is None:
            return False
        if self.config["task"] == "classification":
            return metric > best_metric
        return metric < best_metric

    def _initial_best_metric(self):
        if self.config["task"] == "classification":
            return -np.inf
        return np.inf

    def _metric_label(self):
        return "AUC" if self.config["task"] == "classification" else "RMSE"

    def _stage_title(self):
        stage_name = self.config.get("stage_name") or self.config.get("run_tag") or "train"
        trainable_scope = self.config.get("trainable_scope", "all")
        if stage_name == "stage1":
            return "Step 1/3 - Stage 1 (graph head only)"
        if stage_name == "stage2":
            return "Step 2/3 - Stage 2 (all task heads)"
        if stage_name == "stage3":
            return "Step 3/3 - Stage 3 (full finetune)"
        if stage_name == "single":
            return "Step 1/1 - Single stage"
        return f"{stage_name} ({trainable_scope})"

    def _progress_label(self, split_name):
        trial_serial = self.config.get("trial_serial", "?")
        return f"trial {trial_serial} | {self.config['task_name']} | {self.config.get('stage_name', 'train')} | {split_name}"

    def _batch_postfix(self, split_name: str, running_loss: float, running_parts: Dict[str, float], count: int):
        postfix = {
            f"{split_name}_loss": f"{running_loss / count:.4f}",
            f"{split_name}_graph": f"{running_parts['graph_loss'] / count:.4f}",
        }
        return postfix

    def _loss_part_keys(self):
        return [
            "graph_loss",
            "finger_loss",
            "atom_fg_loss",
            "raw_total_loss",
            "raw_mean_loss",
        ]

    def _empty_eval_result(self):
        parts = {key: np.nan for key in self._loss_part_keys()}
        if self.config["task"] == "classification":
            metrics = {"auc": np.nan}
            return np.nan, np.nan, parts, metrics
        metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
        return np.nan, np.nan, parts, metrics

    def _epoch_metric_value(self, metric_dict: Dict):
        if self.config["task"] == "classification":
            return metric_dict.get("auc")
        return metric_dict.get("rmse")

    def _epoch_metric_key(self):
        return "auc" if self.config["task"] == "classification" else "rmse"

    def _epoch_postfix(self, train_loss, valid_loss, test_loss, train_metrics, valid_metrics, test_metrics):
        metric_key = self._epoch_metric_key()
        return {
            "train_loss": f"{train_loss:.4f}",
            "valid_loss": f"{valid_loss:.4f}",
            "test_loss": f"{test_loss:.4f}",
            f"train_{metric_key}": f"{self._epoch_metric_value(train_metrics):.4f}",
            f"valid_{metric_key}": f"{self._epoch_metric_value(valid_metrics):.4f}",
            f"test_{metric_key}": f"{self._epoch_metric_value(test_metrics):.4f}",
        }

    def _save_best_model(self):
        if RoutineControl.save_best_ckpt:
            torch.save(self._checkpoint_payload(), os.path.join(self.writer.log_dir, "model.pth"))

    def train(self):
        write_record(self.txtfile, self.config)
        self.net = self.net.to("cuda")
        mode = "lower"
        stopper_tmp_path = os.path.join(self.writer.log_dir, "_early_stop_tmp.pth")
        stopper = EarlyStopping(mode=mode, patience=GlobalVar.patience, filename=stopper_tmp_path)

        epoch_list = []
        train_loss_list, valid_loss_list, test_loss_list = [], [], []
        train_metric_list, val_metric_list, test_metric_list = [], [], []
        train_r2_list, val_r2_list, test_r2_list = [], [], []
        mt_weight_history = {"graph": [], "finger": [], "atom_fg": []}
        loss_part_keys = self._loss_part_keys()
        loss_part_history = {
            "train": {key: [] for key in loss_part_keys},
            "valid": {key: [] for key in loss_part_keys},
            "test": {key: [] for key in loss_part_keys},
        }
        best_epoch = None
        best_model_path = os.path.join(self.writer.log_dir, "model.pth")
        stage_title = self._stage_title()
        trial_serial = self.config.get("trial_serial", "?")
        tqdm.write(
            f"\n[Trial {trial_serial}] {stage_title} | "
            f"epochs={self.config['epochs']} | batch_size={self.config['batch_size']} | "
            f"lr={self.config['optim']['init_lr']} | target_transform={self.config.get('regression_label_transform', 'none')}"
        )

        epoch_progress = tqdm(
            range(self.start_epoch, self.config["epochs"] + 1),
            desc=f"trial {trial_serial} | {stage_title}",
            unit="epoch",
            dynamic_ncols=True,
        )
        if self.config.get("eval_before_train", True):
            valid_loss0, valid_metric0, valid_parts0, valid_metrics0 = self._valid_step(
                self.val_loader,
                "valid_epoch0_loaded_checkpoint",
            )
            if self.has_test_set:
                test_loss0, test_metric0, test_parts0, test_metrics0 = self._valid_step(
                    self.test_loader,
                    "test_epoch0_loaded_checkpoint",
                )
            else:
                test_loss0, test_metric0, test_parts0, test_metrics0 = self._empty_eval_result()
            write_record(
                self.txtfile,
                f"epoch:0_loaded_checkpoint\n"
                f"valid_loss:{valid_loss0} test_loss:{test_loss0}\n"
                f"valid_metric:{valid_metric0} test_metric:{test_metric0}\n"
                f"valid_metrics:{valid_metrics0} test_metrics:{test_metrics0}\n"
                f"valid_loss_parts:{valid_parts0} test_loss_parts:{test_parts0}\n"
                f"mt_weights:{self._mt_weights_dict()}",
            )
        for i in epoch_progress:
            if self.config["lr_scheduler"]["type"] in ["cos", "square", "linear"]:
                self.lr_scheduler.adjust_lr(self.optim, i)

            train_loss, train_metric, train_parts, train_metrics = self._train_step()
            valid_loss, valid_metric, valid_parts, valid_metrics = self._valid_step(self.val_loader, "valid")
            if self.has_test_set:
                test_loss, test_metric, test_parts, test_metrics = self._valid_step(self.test_loader, "test")
            else:
                test_loss, test_metric, test_parts, test_metrics = self._empty_eval_result()
            valid_select_metric = self._epoch_metric_value(valid_metrics)

            epoch_list.append(i)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)
            train_metric_list.append(train_metric)
            val_metric_list.append(valid_metric)
            test_metric_list.append(test_metric)
            if self.config["task"] == "regression":
                train_r2_list.append(train_metrics["r2"])
                val_r2_list.append(valid_metrics["r2"])
                test_r2_list.append(test_metrics["r2"])
            for key in loss_part_history["train"]:
                loss_part_history["train"][key].append(train_parts[key])
                loss_part_history["valid"][key].append(valid_parts[key])
                loss_part_history["test"][key].append(test_parts[key])
            mt_weights = self._mt_weights_dict()
            for key in mt_weight_history:
                if mt_weights is None:
                    mt_weight_history[key].append(np.nan)
                else:
                    mt_weight_history[key].append(mt_weights.get(key, np.nan))

            if self._is_better(valid_select_metric, self.best_metric):
                self.best_metric = valid_select_metric
                best_epoch = i
                self._update_best_metadata(
                    i,
                    valid_select_metric,
                    train_metrics,
                    valid_metrics,
                    test_metrics,
                )
                self._save_best_model()
                write_record(
                    self.txtfile,
                    f"best_model_updated epoch:{best_epoch} "
                    f"selection_metric:{self.best_metadata.get('selection_metric')} "
                    f"valid_{self._epoch_metric_key()}:{valid_select_metric} "
                    f"path:{best_model_path}",
                )

            if self.config["task"] == "classification":
                early_stop_score = -valid_metrics["auc"]
                early_stop_test_score = -test_metrics["auc"] if self.has_test_set else np.nan
            else:
                early_stop_score = valid_metrics["rmse"]
                early_stop_test_score = test_metrics["rmse"] if self.has_test_set else np.nan

            if stopper.step(early_stop_score, self.net, test_score=early_stop_test_score):
                stopper.report_final_results(i_epoch=i)
                break

            self.writer.add_scalar("valid_loss", valid_loss, global_step=i)
            self.writer.add_scalar("test_loss", test_loss, global_step=i)
            write_record(
                self.txtfile,
                f"epoch:{i}\n"
                f"train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n"
                f"train_metric:{train_metric} valid_metric:{valid_metric} test_metric:{test_metric}\n"
                f"train_metrics:{train_metrics} valid_metrics:{valid_metrics} test_metrics:{test_metrics}\n"
                f"train_loss_parts:{train_parts} valid_loss_parts:{valid_parts} test_loss_parts:{test_parts}\n"
                f"mt_weights:{self._mt_weights_dict()}",
            )
            epoch_progress.set_postfix(
                self._epoch_postfix(
                    train_loss, valid_loss, test_loss,
                    train_metrics, valid_metrics, test_metrics,
                )
            )
        if best_epoch is None and val_metric_list:
            if self.config["task"] == "classification":
                best_idx = int(np.argmax(val_metric_list))
            else:
                best_idx = int(np.argmin(val_metric_list))
            best_epoch = epoch_list[best_idx]
        self._save_learning_curve(
            epoch_list, train_loss_list, valid_loss_list, test_loss_list,
            train_metric_list, val_metric_list, test_metric_list,
            loss_part_history, mt_weight_history, best_epoch, best_model_path,
            train_r2_list, val_r2_list, test_r2_list,
        )
        write_record(self.txtfile, f"best_epoch:{best_epoch}\tbest_model_path:{best_model_path}")
        metric_key = self._epoch_metric_key()
        best_idx = epoch_list.index(best_epoch)
        train_best_metric = train_metric_list[best_idx]
        valid_best_metric = val_metric_list[best_idx]
        test_best_metric = test_metric_list[best_idx]
        test_metric_text = f"{test_best_metric:.6f}" if self.has_test_set else "N/A"
        tqdm.write(
            f"[Trial {trial_serial}] {stage_title} completed | "
            f"best_epoch={best_epoch} | "
            f"train_{metric_key}={train_best_metric:.6f} | "
            f"valid_{metric_key}={valid_best_metric:.6f} | "
            f"test_{metric_key}={test_metric_text} | "
            f"model={best_model_path}"
        )
        if os.path.exists(stopper_tmp_path):
            os.remove(stopper_tmp_path)

    def _save_learning_curve(self, epoch_list, train_loss_list, valid_loss_list, test_loss_list,
                             train_metric_list, val_metric_list, test_metric_list,
                             loss_part_history, mt_weight_history, best_epoch, best_model_path,
                             train_r2_list, val_r2_list, test_r2_list):
        if not epoch_list:
            return
        save_path = os.path.join(self.writer.log_dir, "learning_curve.png")
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        font_size = 7
        title_size = 9
        legend_size = 6
        text_size = 6
        line_width = 1.0
        marker_line_width = 0.8
        has_test_curve = not all(np.isnan(value) for value in test_metric_list)
        has_test_loss_curve = not all(np.isnan(value) for value in test_loss_list)

        ax = axes[0, 0]
        ax.plot(epoch_list, train_loss_list, label="train_dynamic_mt_loss", linewidth=line_width)
        ax.plot(epoch_list, valid_loss_list, label="valid_dynamic_mt_loss", linewidth=line_width)
        if has_test_loss_curve:
            ax.plot(epoch_list, test_loss_list, label="test_dynamic_mt_loss", linewidth=line_width)
        ax.plot(
            epoch_list,
            loss_part_history["train"]["raw_mean_loss"],
            linestyle="--",
            label="train_raw_mean_loss",
            linewidth=line_width,
        )
        ax.plot(
            epoch_list,
            loss_part_history["valid"]["raw_mean_loss"],
            linestyle="--",
            label="valid_raw_mean_loss",
            linewidth=line_width,
        )
        if has_test_loss_curve:
            ax.plot(
                epoch_list,
                loss_part_history["test"]["raw_mean_loss"],
                linestyle="--",
                label="test_raw_mean_loss",
                linewidth=line_width,
            )
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width, label=f"best_epoch={best_epoch}")
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.set_ylabel("Loss value", fontsize=font_size)
        ax.set_title("Dynamic MT Loss vs Raw Mean Loss", fontsize=title_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size, ncol=2)

        ax = axes[0, 1]
        for split_name, style in [("train", "-"), ("valid", "--"), ("test", ":")]:
            if split_name == "test" and not has_test_loss_curve:
                continue
            for part_name in ["graph_loss", "finger_loss", "atom_fg_loss", "raw_total_loss"]:
                ax.plot(epoch_list, loss_part_history[split_name][part_name],
                        linestyle=style, label=f"{split_name}_{part_name}", linewidth=line_width)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width)
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.set_ylabel("Raw loss component", fontsize=font_size)
        ax.set_title("Raw Loss Components", fontsize=title_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size, ncol=2)

        ax = axes[1, 0]
        primary_label = "RMSE" if self.config["task"] == "regression" else "AUC"
        ax.plot(epoch_list, train_metric_list, label=f"train_{primary_label}", linewidth=line_width)
        ax.plot(epoch_list, val_metric_list, label=f"valid_{primary_label}", linewidth=line_width)
        if has_test_curve:
            ax.plot(epoch_list, test_metric_list, label=f"test_{primary_label}", linewidth=line_width)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width)
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.set_ylabel(primary_label, fontsize=font_size)
        if self.config["task"] == "regression":
            ax.set_title("RMSE Curve: model selection uses valid_RMSE", fontsize=title_size)
        else:
            ax.set_title(f"{primary_label} Curve", fontsize=title_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size)

        ax = axes[1, 1]
        if self.config["task"] == "regression":
            ax.plot(epoch_list, train_r2_list, label="train_R2", linewidth=line_width)
            ax.plot(epoch_list, val_r2_list, label="valid_R2", linewidth=line_width)
            if has_test_curve:
                ax.plot(epoch_list, test_r2_list, label="test_R2", linewidth=line_width)
            ax.set_ylabel("R2", fontsize=font_size)
            ax.set_title("R2 Curve: reference only", fontsize=title_size)
        else:
            gap = np.array(valid_loss_list) - np.array(train_loss_list)
            ax.plot(epoch_list, gap, label="valid_loss - train_loss", linewidth=line_width)
            ax.set_ylabel("Loss Gap", fontsize=font_size)
            ax.set_title("Generalization Gap", fontsize=title_size)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width)
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size)

        ax = axes[2, 0]
        for key in ["graph", "finger", "atom_fg"]:
            ax.plot(epoch_list, mt_weight_history[key], label=f"mt_weight_{key}", linewidth=line_width)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width)
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.set_ylabel("Dynamic MT weight", fontsize=font_size)
        ax.set_title("Dynamic Multi-task Weights", fontsize=title_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size)

        ax = axes[2, 1]
        if self.config["task"] == "regression":
            gap = np.array(val_metric_list) - np.array(train_metric_list)
            ax.plot(epoch_list, gap, label="valid_RMSE - train_RMSE", linewidth=line_width)
            ax.set_ylabel("RMSE gap", fontsize=font_size)
            ax.set_title("Generalization Gap", fontsize=title_size)
        else:
            gap = np.array(valid_loss_list) - np.array(train_loss_list)
            ax.plot(epoch_list, gap, label="valid_loss - train_loss", linewidth=line_width)
            ax.set_ylabel("Loss gap", fontsize=font_size)
            ax.set_title("Generalization Gap", fontsize=title_size)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", color="black", linewidth=marker_line_width)
        ax.text(0.02, 0.02, f"best_model_path:\n{best_model_path}",
                transform=ax.transAxes, fontsize=text_size, verticalalignment="bottom")
        ax.set_xlabel("Epoch", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=legend_size)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"learning_curve_saved:{save_path}")


def apply_dist_bar(dist_bar):
    if len(dist_bar) == 1:
        GlobalVar.dist_bar = [dist_bar[0]]
    elif len(dist_bar) == 2:
        set_dist_bar_two(dist_bar[0], dist_bar[1])
    elif len(dist_bar) == 3:
        set_dist_bar_three(dist_bar[0], dist_bar[1], dist_bar[2])
    else:
        raise ValueError(f"dist_bar must contain 1, 2, or 3 thresholds, got: {dist_bar}")
    if GlobalVar.dist_bar is None:
        raise ValueError(f"Invalid dist_bar thresholds: {dist_bar}")


def normalize_dist_bar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def dist_bar_grid(search_value, default_value):
    value = default_value if search_value is None else search_value
    if not isinstance(value, list):
        return [[int(value)]]
    if not value:
        raise ValueError("dist_bar cannot be empty.")
    if all(isinstance(item, list) for item in value):
        return [normalize_dist_bar(item) for item in value]
    if all(not isinstance(item, list) for item in value):
        return [normalize_dist_bar(value)]
    raise ValueError("dist_bar search values must be like [30, 60] or [[20, 60], [30, 60]].")


def _normalize_for_signature(value):
    if isinstance(value, dict):
        return {key: _normalize_for_signature(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(item) for item in value]
    return value


def _effective_trial_signature(trial):
    sig = dict(trial)
    if sig.get("regression_label_transform") != "winsorized_standardize":
        sig.pop("winsorize_quantile_low", None)
        sig.pop("winsorize_quantile_high", None)
    if not sig.get("graph_label_reweight", False):
        sig.pop("graph_label_bins", None)
        sig.pop("graph_label_weight_power", None)
        sig.pop("graph_label_weight_clip", None)
    sig.pop("run_tag_base", None)
    return json.dumps(
        _normalize_for_signature(sig),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def build_config(base_args, trial, trainable_scope, checkpoint):
    config_path = Path(pdir) / "config" / "config_finetune.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(base_args.gpus)
    GlobalVar.use_ckpt = bool(checkpoint)

    user = "mpp"
    config["userconfig"][user]["dataset_dir"] = base_args.dataroot
    config["userconfig"][user]["split_dir"] = base_args.splitroot
    split_pkl_path = trial.get("split_pkl_path", getattr(base_args, "split_pkl_path", None))
    if split_pkl_path:
        split_pkl_path = resolve_path(Path(__file__).resolve().parent, split_pkl_path)
        if not split_pkl_path.exists():
            raise FileNotFoundError(f"split_pkl_path not found: {split_pkl_path}")
        split_pkl_path = str(split_pkl_path)
        config["userconfig"][user]["split_file"] = split_pkl_path
    config = config_current_user(user, config)
    config = config_dataset_form("pkl", config)
    config["split_type"] = base_args.split_type
    config["split_pkl_path"] = split_pkl_path
    config["allow_empty_test"] = bool(trial.get("allow_empty_test", getattr(base_args, "allow_empty_test", False)))
    config["task_name"] = base_args.task
    config["epochs"] = trial["epochs"]
    config["batch_size"] = trial["batch_size"]
    config["seed"] = trial["seed"]
    config["dropout"] = trial["dropout"]
    DEFAULTS.DROP_RATE = trial["dropout"]
    config["save_ckpt"] = max(1, int(base_args.save_ckpt))
    config["dataloader_num_workers"] = base_args.dataloader_num_workers
    config["pin_memory"] = bool(base_args.pin_memory)
    config["optim"]["init_lr"] = trial["lr"]
    config["optim"]["init_base_lr"] = trial.get("init_base_lr", config["optim"].get("init_base_lr", 1e-4))
    config["optim"]["weight_decay"] = trial["weight_decay"]
    config["optim"]["type"] = trial.get("optim_type", base_args.optim_type)
    config["optim"]["momentum"] = 0
    config["lr_scheduler"]["type"] = trial.get("scheduler_type", base_args.scheduler_type)
    config["lr_scheduler"]["warm_up_epoch"] = trial.get("warm_up_epoch", base_args.warm_up_epoch)
    config["lr_scheduler"]["start_lr"] = trial.get("start_lr", config["lr_scheduler"].get("start_lr", 1e-5))
    config["checkpoint"] = checkpoint
    config["run_tag"] = trial["run_tag"]
    config["log_dir"] = trial.get("log_dir")
    config["trial_dir"] = trial.get("trial_dir")
    config["trial_serial"] = trial.get("trial_serial")
    config["stage_name"] = trial.get("stage_name")
    config["trainable_scope"] = trainable_scope
    config["loss_weights"] = trial.get("loss_weights")
    config["regression_label_transform"] = trial.get("regression_label_transform", "none")
    config["encoder_lr_ratio"] = trial.get("encoder_lr_ratio", 1.0)
    config["max_grad_norm"] = trial.get("max_grad_norm", 0.0)
    config["eval_before_train"] = trial.get("eval_before_train", True)
    config["mt_loss_eps"] = trial.get("mt_loss_eps", 1e-8)
    config["mt_loss_tau"] = trial.get("mt_loss_tau", 2.0)
    config["mt_ratio_clip"] = trial.get("mt_ratio_clip", 5.0)
    config["mt_loss_prior"] = trial.get("mt_loss_prior", None)
    config["winsorize_quantile_low"] = trial.get("winsorize_quantile_low", 0.01)
    config["winsorize_quantile_high"] = trial.get("winsorize_quantile_high", 0.99)
    config["graph_label_reweight"] = trial.get("graph_label_reweight", False)
    config["graph_label_bins"] = trial.get(
        "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
    )
    config["graph_label_weight_power"] = trial.get("graph_label_weight_power", 0.5)
    config["graph_label_weight_clip"] = trial.get("graph_label_weight_clip", 3.0)
    config["pretrain_model_path"] = "None"
    config = get_downstream_task_names(config)
    set_seed(config["seed"])
    GlobalVar.patience = trial["patience"]
    config["patience"] = GlobalVar.patience
    RoutineControl.save_best_ckpt = True
    apply_dist_bar(trial.get("dist_bar", base_args.dist_bar))
    GlobalVar.embedding_style = "more"
    GlobalVar.transformer_dim = base_args.embedding_dim
    GlobalVar.ffn_dim = base_args.hidden_dim
    GlobalVar.num_heads = base_args.num_heads
    GlobalVar.freeze_layers = 0
    GlobalVar.parallel_train = False
    config["model"]["atom_embed_dim"] = GlobalVar.transformer_dim
    config["model"]["bond_embed_dim"] = GlobalVar.transformer_dim
    config["model"]["hidden_size"] = GlobalVar.ffn_dim
    config["model"]["num_heads"] = GlobalVar.num_heads
    config["model"]["layer_num"] = base_args.layer_num
    config["fg_num_"] = nfg() + 1
    config["freeze_layers"] = GlobalVar.freeze_layers
    config["loss_style"] = GlobalVar.loss_style

    upstream.config = config
    upstream.path = config_path
    upstream.get_finetune_log_dir = custom_log_dir
    return config, config_path


def run_training(base_args, trial, trainable_scope, checkpoint):
    config, config_path = build_config(base_args, trial, trainable_scope, checkpoint)
    trainer = CustomTrainer(config, config_path)
    trainer.train()
    model_path = Path(trainer.writer.log_dir) / "model.pth"
    if hasattr(trainer, "writer"):
        trainer.writer.close()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Training finished but no best checkpoint was saved: {model_path}. "
            "Check the stage record.txt for non-finite validation metrics."
        )
    if int(base_args.minimal_outputs) == 1:
        keep_names = {"model.pth", "config_finetune.yaml", "learning_curve.png", "record.txt"}
        for artifact in Path(model_path).parent.iterdir():
            if artifact.name in keep_names:
                continue
            if artifact.is_dir():
                shutil.rmtree(artifact, ignore_errors=True)
            else:
                try:
                    artifact.unlink()
                except FileNotFoundError:
                    pass
    return model_path


def build_trials(cfg, default_dist_bar):
    fixed = cfg.get("fixed", {}) or {}
    search = cfg.get("search", cfg.get("grid", {})) or {}
    three_stage = cfg.get("three_stage", cfg.get("two_stage", {})) or {}
    base = {
        "seed": fixed.get("seed", 8),
        "batch_size": fixed.get("batch_size", 16),
        "dropout": fixed.get("dropout", 0.1),
        "weight_decay": fixed.get("weight_decay", 1e-4),
        "init_base_lr": fixed.get("init_base_lr", 1e-4),
        "optim_type": fixed.get("optim_type", "adam"),
        "warm_up_epoch": fixed.get("warm_up_epoch", 5),
        "start_lr": fixed.get("start_lr", 1e-5),
        "stage1_lr": three_stage.get("stage1_lr", fixed.get("lr", 5e-5)),
        "stage2_lr": three_stage.get("stage2_lr", fixed.get("lr", 5e-5)),
        "stage3_lr": three_stage.get("stage3_lr", fixed.get("lr", 5e-5)),
        "stage1_epochs": three_stage.get("stage1_epochs", 20),
        "stage2_epochs": three_stage.get("stage2_epochs", 20),
        "stage3_epochs": three_stage.get("stage3_epochs", fixed.get("epochs", 100)),
        "stage1_patience": three_stage.get("stage1_patience", 10),
        "stage2_patience": three_stage.get("stage2_patience", 10),
        "stage3_patience": three_stage.get("stage3_patience", fixed.get("patience", 30)),
        "scheduler_type": fixed.get("scheduler_type", "None"),
        "regression_label_transform": fixed.get("regression_label_transform", "none"),
        "encoder_lr_ratio": fixed.get("encoder_lr_ratio", 0.05),
        "max_grad_norm": fixed.get("max_grad_norm", 1.0),
        "eval_before_train": fixed.get("eval_before_train", True),
        "mt_loss_eps": fixed.get("mt_loss_eps", 1e-8),
        "mt_loss_tau": fixed.get("mt_loss_tau", 2.0),
        "mt_ratio_clip": fixed.get("mt_ratio_clip", 5.0),
        "mt_loss_prior": fixed.get("mt_loss_prior", None),
        "winsorize_quantile_low": fixed.get("winsorize_quantile_low", 0.01),
        "winsorize_quantile_high": fixed.get("winsorize_quantile_high", 0.99),
        "graph_label_reweight": fixed.get("graph_label_reweight", False),
        "graph_label_bins": fixed.get(
            "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
        ),
        "graph_label_weight_power": fixed.get("graph_label_weight_power", 0.5),
        "graph_label_weight_clip": fixed.get("graph_label_weight_clip", 3.0),
        "stage2_warmup_epochs": three_stage.get("stage2_warmup_epochs", fixed.get("warm_up_epoch", 5)),
        "stage3_warmup_epochs": three_stage.get("stage3_warmup_epochs", fixed.get("warm_up_epoch", 5)),
        "stage1_loss_weights": fixed.get(
            "stage1_loss_weights",
            {"graph": 1.0, "finger": 0.0, "atom_fg": 0.0},
        ),
        "stage2_loss_weights": fixed.get("stage2_loss_weights", None),
        "stage3_loss_weights": fixed.get("stage3_loss_weights", fixed.get("loss_weights")),
        "split_pkl_path": fixed.get("split_pkl_path", cfg.get("split_pkl_path")),
        "allow_empty_test": fixed.get("allow_empty_test", cfg.get("allow_empty_test", False)),
        "dist_bar": fixed.get("dist_bar", default_dist_bar),
    }
    grids = {key: as_list(search.get(key), base[key]) for key in base if key != "dist_bar"}
    grids["dist_bar"] = dist_bar_grid(search.get("dist_bar"), base["dist_bar"])
    trials = []
    for idx, values in enumerate(itertools.product(*(grids[key] for key in base)), start=1):
        item = dict(zip(base.keys(), values))
        tag = (
            f"grid{idx}_seed{item['seed']}_bs{item['batch_size']}_do{item['dropout']}"
            f"_wd{item['weight_decay']}_opt{item['optim_type']}_wu{item['warm_up_epoch']}"
            f"_s1lr{item['stage1_lr']}_s2lr{item['stage2_lr']}_s3lr{item['stage3_lr']}"
            f"_elr{item['encoder_lr_ratio']}"
            f"_sch{item['scheduler_type']}_yt{item['regression_label_transform']}"
            f"_db{'-'.join(str(x) for x in item['dist_bar'])}"
        ).replace(".", "p")
        item["run_tag_base"] = tag
        trials.append(item)

    deduped = []
    seen = set()
    for trial in trials:
        sig = _effective_trial_signature(trial)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(trial)
    print(
        f"[Search] raw trials={len(trials)} "
        f"deduped trials={len(deduped)} "
        f"removed={len(trials) - len(deduped)}"
    )
    return deduped


def eval_settings(args, trial):
    splitroot = Path(args.splitroot)
    split_root = splitroot.parent if splitroot.name == "split" else splitroot.parent
    return {
        "task": args.task,
        "split_type": args.split_type,
        "seed": trial["seed"],
        "batch_size": trial["batch_size"],
        "dataloader_num_workers": args.dataloader_num_workers,
        "gpus": args.gpus,
        "result_dir": args.result_dir,
        "pkl_dir": args.dataroot,
        "split_root": str(split_root),
        "split_pkl_path": trial.get("split_pkl_path", getattr(args, "split_pkl_path", None)),
        "allow_empty_test": bool(trial.get("allow_empty_test", getattr(args, "allow_empty_test", False))),
    }


def try_resume_trial_from_disk(
    idx: int,
    trial_cfg: Dict,
    trial_dir: Path,
    args,
    project_root: Path,
    py: str,
    pkl_dir: Path,
    split_root: Path,
):
    """Rebuild one trial candidate from checkpoints if training already finished on disk."""
    trial_dir = Path(trial_dir)
    eval_args = settings_namespace(eval_settings(args, trial_cfg))

    if args.single_stage:
        candidate_model = trial_dir / "single" / "model.pth"
        if not candidate_model.is_file():
            return None
        stage1_model = stage2_model = None
        stage1_valid_metric = stage1_test_metric = None
        stage2_valid_metric = stage2_test_metric = None
        stage1_tag = stage2_tag = stage3_tag = None
        final_tag = "single"
        valid_metric = run_predict_eval_metrics_only(
            py, project_root, eval_args, pkl_dir, split_root, candidate_model, "valid"
        )
        test_metric = run_predict_eval_metrics_only(
            py, project_root, eval_args, pkl_dir, split_root, candidate_model, "test"
        )
        return {
            "idx": idx,
            "trial_dir": str(trial_dir),
            "run_tag": final_tag,
            "model": str(candidate_model),
            "stage1_run_tag": stage1_tag,
            "stage2_run_tag": stage2_tag,
            "stage3_run_tag": stage3_tag,
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            "stage3_model": None,
            "params": {**trial_cfg, "single_stage": args.single_stage},
            "stage1_valid": stage1_valid_metric,
            "stage1_test": stage1_test_metric,
            "stage2_valid": stage2_valid_metric,
            "stage2_test": stage2_test_metric,
            "valid": valid_metric,
            "test": test_metric,
        }

    stage1_model = trial_dir / "stage1" / "model.pth"
    stage2_model = trial_dir / "stage2" / "model.pth"
    stage3_model = trial_dir / "stage3" / "model.pth"
    if not (stage1_model.is_file() and stage2_model.is_file() and stage3_model.is_file()):
        return None

    stage1_tag = "stage1"
    stage2_tag = "stage2"
    stage3_tag = "stage3"
    final_tag = stage3_tag
    candidate_model = stage3_model

    stage1_valid_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, stage1_model, "valid"
    )
    stage1_test_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, stage1_model, "test"
    )
    stage2_valid_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, stage2_model, "valid"
    )
    stage2_test_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, stage2_model, "test"
    )
    valid_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, candidate_model, "valid"
    )
    test_metric = run_predict_eval_metrics_only(
        py, project_root, eval_args, pkl_dir, split_root, candidate_model, "test"
    )

    return {
        "idx": idx,
        "trial_dir": str(trial_dir),
        "run_tag": final_tag,
        "model": str(candidate_model),
        "stage1_run_tag": stage1_tag,
        "stage2_run_tag": stage2_tag,
        "stage3_run_tag": stage3_tag,
        "stage1_model": str(stage1_model),
        "stage2_model": str(stage2_model),
        "stage3_model": str(candidate_model),
        "params": {**trial_cfg, "single_stage": args.single_stage},
        "stage1_valid": stage1_valid_metric,
        "stage1_test": stage1_test_metric,
        "stage2_valid": stage2_valid_metric,
        "stage2_test": stage2_test_metric,
        "valid": valid_metric,
        "test": test_metric,
    }


def metric_is_better(task_type, new_metric, best_metric):
    if new_metric is None:
        return False
    if best_metric is None:
        return True
    if task_type == "regression":
        return new_metric < best_metric
    return new_metric > best_metric


def predict_primary_metric_name(metric):
    if metric["task_type"] == "classification":
        return "roc_auc" if "roc_auc" in metric else "auc"
    return "rmse"


def metric_value_text(metric, metric_name):
    if not metric or metric_name is None:
        return "N/A"
    value = metric.get(metric_name)
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value:.6f}"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    py = sys.executable
    file_cfg = read_search_config(args.search_config)
    cfg_tag = config_name_tag(args.search_config)
    for key in [
        "task",
        "dataroot",
        "splitroot",
        "split_type",
        "split_pkl_path",
        "allow_empty_test",
        "result_dir",
        "gpus",
        "dist_bar",
    ]:
        if key in file_cfg:
            setattr(args, key, file_cfg[key])

    pkl_dir = resolve_path(project_root, args.dataroot)
    splitroot = resolve_path(project_root, args.splitroot)
    split_root = splitroot.parent if splitroot.name == "split" else splitroot.parent
    result_dir = resolve_path(project_root, args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    best_model = None
    best_metric = None
    best_candidate = None

    trials = build_trials(file_cfg, args.dist_bar)
    trial_progress = tqdm(trials, desc=f"{args.task} hyperparameter trials", unit="trial", dynamic_ncols=True)
    for idx, trial_cfg in enumerate(trial_progress, start=1):
        trial_name = trial_folder_name(args.task, args.split_type, cfg_tag, idx)
        trial_dir = project_root / "finetune_result" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg["trial_serial"] = idx
        trial_cfg["trial_name"] = trial_name
        trial_cfg["trial_dir"] = str(trial_dir)
        trial_progress.set_postfix(
            trial=trial_name,
            transform=trial_cfg.get("regression_label_transform", "none"),
        )
        tqdm.write(f"\n[Trial {idx}/{len(trials)}] {trial_name} started")
        eval_args = settings_namespace(eval_settings(args, trial_cfg))
        if args.resume_disk:
            resumed_candidate = try_resume_trial_from_disk(
                idx, trial_cfg, trial_dir, args, project_root, py, pkl_dir, split_root
            )
            if resumed_candidate is not None:
                valid_metric = resumed_candidate["valid"]
                test_metric = resumed_candidate["test"]
                final_metric_name = predict_primary_metric_name(valid_metric)
                tqdm.write(
                    f"[Trial {idx}/{len(trials)}] resume_disk | re-eval | "
                    f"valid_{final_metric_name}={valid_metric.get(final_metric_name):.6f} | "
                    f"test_{final_metric_name}={metric_value_text(test_metric, final_metric_name)} | "
                    f"model={resumed_candidate['model']}"
                )
                task_type = valid_metric["task_type"]
                primary_name = predict_primary_metric_name(valid_metric)
                primary_value = valid_metric.get(primary_name)
                candidates.append(resumed_candidate)
                display_metric_name = primary_name
                display_metric_value = primary_value
                tqdm.write(
                    f"[Trial {idx}/{len(trials)}] completed | "
                    f"valid_{display_metric_name}={display_metric_value:.6f} | model={resumed_candidate['model']}"
                )
                if metric_is_better(task_type, primary_value, best_metric):
                    best_metric = primary_value
                    best_model = Path(resumed_candidate["model"])
                    best_candidate = resumed_candidate
                    tqdm.write(
                        f"[Trial {idx}/{len(trials)}] new best | "
                        f"valid_{display_metric_name}={display_metric_value:.6f}"
                    )
                continue
        stage1_model = None
        stage1_valid_metric = None
        stage1_test_metric = None
        stage1_tag = None
        stage2_model = None
        stage2_valid_metric = None
        stage2_test_metric = None
        stage2_tag = None
        stage3_tag = None

        if args.single_stage:
            final_tag = "single"
            stage2_trial = {
                "seed": trial_cfg["seed"],
                "batch_size": trial_cfg["batch_size"],
                "dropout": trial_cfg["dropout"],
                "weight_decay": trial_cfg["weight_decay"],
                "init_base_lr": trial_cfg["init_base_lr"],
                "optim_type": trial_cfg["optim_type"],
                "warm_up_epoch": trial_cfg.get("stage3_warmup_epochs", trial_cfg["warm_up_epoch"]),
                "start_lr": trial_cfg["start_lr"],
                "lr": trial_cfg["stage3_lr"],
                "epochs": trial_cfg["stage3_epochs"],
                "patience": trial_cfg["stage3_patience"],
                "scheduler_type": trial_cfg["scheduler_type"],
                "run_tag": final_tag,
                "log_dir": str(trial_dir / "single"),
                "trial_dir": str(trial_dir),
                "trial_serial": idx,
                "stage_name": "single",
                "loss_weights": trial_cfg.get("stage3_loss_weights"),
                "encoder_lr_ratio": trial_cfg.get("encoder_lr_ratio", 0.05),
                "max_grad_norm": trial_cfg.get("max_grad_norm", 1.0),
                "eval_before_train": trial_cfg.get("eval_before_train", True),
                "mt_loss_eps": trial_cfg.get("mt_loss_eps", 1e-8),
                "mt_loss_tau": trial_cfg.get("mt_loss_tau", 2.0),
                "mt_ratio_clip": trial_cfg.get("mt_ratio_clip", 5.0),
                "mt_loss_prior": trial_cfg.get("mt_loss_prior", None),
                "regression_label_transform": trial_cfg.get("regression_label_transform", "none"),
                "winsorize_quantile_low": trial_cfg.get("winsorize_quantile_low", 0.01),
                "winsorize_quantile_high": trial_cfg.get("winsorize_quantile_high", 0.99),
                "graph_label_reweight": trial_cfg.get("graph_label_reweight", False),
                "graph_label_bins": trial_cfg.get(
                    "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
                ),
                "graph_label_weight_power": trial_cfg.get("graph_label_weight_power", 0.5),
                "graph_label_weight_clip": trial_cfg.get("graph_label_weight_clip", 3.0),
                "dist_bar": trial_cfg["dist_bar"],
                "split_pkl_path": trial_cfg.get("split_pkl_path"),
                "allow_empty_test": trial_cfg.get("allow_empty_test", False),
            }
            save_trial_params(trial_dir, args, trial_cfg, {"single": stage2_trial})
            candidate_model = run_training(args, stage2_trial, "all", args.pretrain_ckpt)
        else:
            stage1_tag = "stage1"
            stage1_trial = {
                "seed": trial_cfg["seed"],
                "batch_size": trial_cfg["batch_size"],
                "dropout": trial_cfg["dropout"],
                "weight_decay": trial_cfg["weight_decay"],
                "init_base_lr": trial_cfg["init_base_lr"],
                "optim_type": trial_cfg["optim_type"],
                "warm_up_epoch": trial_cfg["warm_up_epoch"],
                "start_lr": trial_cfg["start_lr"],
                "lr": trial_cfg["stage1_lr"],
                "epochs": trial_cfg["stage1_epochs"],
                "patience": trial_cfg["stage1_patience"],
                "scheduler_type": trial_cfg["scheduler_type"],
                "run_tag": stage1_tag,
                "log_dir": str(trial_dir / "stage1"),
                "trial_dir": str(trial_dir),
                "trial_serial": idx,
                "stage_name": "stage1",
                "loss_weights": trial_cfg.get(
                    "stage1_loss_weights",
                    {"graph": 1.0, "finger": 0.0, "atom_fg": 0.0},
                ),
                "encoder_lr_ratio": trial_cfg.get("encoder_lr_ratio", 0.05),
                "max_grad_norm": trial_cfg.get("max_grad_norm", 1.0),
                "eval_before_train": trial_cfg.get("eval_before_train", True),
                "mt_loss_eps": trial_cfg.get("mt_loss_eps", 1e-8),
                "mt_loss_tau": trial_cfg.get("mt_loss_tau", 2.0),
                "mt_ratio_clip": trial_cfg.get("mt_ratio_clip", 5.0),
                "mt_loss_prior": trial_cfg.get("mt_loss_prior", None),
                "regression_label_transform": trial_cfg.get("regression_label_transform", "none"),
                "winsorize_quantile_low": trial_cfg.get("winsorize_quantile_low", 0.01),
                "winsorize_quantile_high": trial_cfg.get("winsorize_quantile_high", 0.99),
                "graph_label_reweight": trial_cfg.get("graph_label_reweight", False),
                "graph_label_bins": trial_cfg.get(
                    "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
                ),
                "graph_label_weight_power": trial_cfg.get("graph_label_weight_power", 0.5),
                "graph_label_weight_clip": trial_cfg.get("graph_label_weight_clip", 3.0),
                "dist_bar": trial_cfg["dist_bar"],
                "split_pkl_path": trial_cfg.get("split_pkl_path"),
                "allow_empty_test": trial_cfg.get("allow_empty_test", False),
            }
            save_trial_params(trial_dir, args, trial_cfg, {"stage1": stage1_trial})
            stage1_model = run_training(args, stage1_trial, "head", args.pretrain_ckpt)
            stage1_valid_metric = run_predict_eval_metrics_only(
                py, project_root, eval_args, pkl_dir, split_root, stage1_model, "valid"
            )
            stage1_test_metric = run_predict_eval_metrics_only(
                py, project_root, eval_args, pkl_dir, split_root, stage1_model, "test"
            )
            stage1_metric_name = predict_primary_metric_name(stage1_valid_metric)
            tqdm.write(
                f"[Trial {idx}/{len(trials)}] stage1 best checkpoint re-eval | "
                f"valid_{stage1_metric_name}={stage1_valid_metric.get(stage1_metric_name):.6f} | "
                f"test_{stage1_metric_name}={metric_value_text(stage1_test_metric, stage1_metric_name)} | "
                f"model={stage1_model}"
            )

            stage2_tag = "stage2"
            stage2_trial = {
                "seed": trial_cfg["seed"],
                "batch_size": trial_cfg["batch_size"],
                "dropout": trial_cfg["dropout"],
                "weight_decay": trial_cfg["weight_decay"],
                "init_base_lr": trial_cfg["init_base_lr"],
                "optim_type": trial_cfg["optim_type"],
                "warm_up_epoch": trial_cfg.get("stage2_warmup_epochs", trial_cfg["warm_up_epoch"]),
                "start_lr": trial_cfg["start_lr"],
                "lr": trial_cfg["stage2_lr"],
                "epochs": trial_cfg["stage2_epochs"],
                "patience": trial_cfg["stage2_patience"],
                "scheduler_type": trial_cfg["scheduler_type"],
                "run_tag": stage2_tag,
                "log_dir": str(trial_dir / "stage2"),
                "trial_dir": str(trial_dir),
                "trial_serial": idx,
                "stage_name": "stage2",
                "loss_weights": trial_cfg.get("stage2_loss_weights"),
                "encoder_lr_ratio": trial_cfg.get("encoder_lr_ratio", 0.05),
                "max_grad_norm": trial_cfg.get("max_grad_norm", 1.0),
                "eval_before_train": trial_cfg.get("eval_before_train", True),
                "mt_loss_eps": trial_cfg.get("mt_loss_eps", 1e-8),
                "mt_loss_tau": trial_cfg.get("mt_loss_tau", 2.0),
                "mt_ratio_clip": trial_cfg.get("mt_ratio_clip", 5.0),
                "mt_loss_prior": trial_cfg.get("mt_loss_prior", None),
                "regression_label_transform": trial_cfg.get("regression_label_transform", "none"),
                "winsorize_quantile_low": trial_cfg.get("winsorize_quantile_low", 0.01),
                "winsorize_quantile_high": trial_cfg.get("winsorize_quantile_high", 0.99),
                "graph_label_reweight": trial_cfg.get("graph_label_reweight", False),
                "graph_label_bins": trial_cfg.get(
                    "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
                ),
                "graph_label_weight_power": trial_cfg.get("graph_label_weight_power", 0.5),
                "graph_label_weight_clip": trial_cfg.get("graph_label_weight_clip", 3.0),
                "dist_bar": trial_cfg["dist_bar"],
                "split_pkl_path": trial_cfg.get("split_pkl_path"),
                "allow_empty_test": trial_cfg.get("allow_empty_test", False),
            }
            save_trial_params(trial_dir, args, trial_cfg, {"stage1": stage1_trial, "stage2": stage2_trial})
            stage2_model = run_training(args, stage2_trial, "heads", stage1_model)
            stage2_valid_metric = run_predict_eval_metrics_only(
                py, project_root, eval_args, pkl_dir, split_root, stage2_model, "valid"
            )
            stage2_test_metric = run_predict_eval_metrics_only(
                py, project_root, eval_args, pkl_dir, split_root, stage2_model, "test"
            )
            stage2_metric_name = predict_primary_metric_name(stage2_valid_metric)
            tqdm.write(
                f"[Trial {idx}/{len(trials)}] stage2 best checkpoint re-eval | "
                f"valid_{stage2_metric_name}={stage2_valid_metric.get(stage2_metric_name):.6f} | "
                f"test_{stage2_metric_name}={metric_value_text(stage2_test_metric, stage2_metric_name)} | "
                f"model={stage2_model}"
            )

            stage3_tag = "stage3"
            final_tag = stage3_tag
            stage3_trial = {
                "seed": trial_cfg["seed"],
                "batch_size": trial_cfg["batch_size"],
                "dropout": trial_cfg["dropout"],
                "weight_decay": trial_cfg["weight_decay"],
                "init_base_lr": trial_cfg["init_base_lr"],
                "optim_type": trial_cfg["optim_type"],
                "warm_up_epoch": trial_cfg.get("stage3_warmup_epochs", trial_cfg["warm_up_epoch"]),
                "start_lr": trial_cfg["start_lr"],
                "lr": trial_cfg["stage3_lr"],
                "epochs": trial_cfg["stage3_epochs"],
                "patience": trial_cfg["stage3_patience"],
                "scheduler_type": trial_cfg["scheduler_type"],
                "run_tag": stage3_tag,
                "log_dir": str(trial_dir / "stage3"),
                "trial_dir": str(trial_dir),
                "trial_serial": idx,
                "stage_name": "stage3",
                "loss_weights": trial_cfg.get("stage3_loss_weights"),
                "encoder_lr_ratio": trial_cfg.get("encoder_lr_ratio", 0.05),
                "max_grad_norm": trial_cfg.get("max_grad_norm", 1.0),
                "eval_before_train": trial_cfg.get("eval_before_train", True),
                "mt_loss_eps": trial_cfg.get("mt_loss_eps", 1e-8),
                "mt_loss_tau": trial_cfg.get("mt_loss_tau", 2.0),
                "mt_ratio_clip": trial_cfg.get("mt_ratio_clip", 5.0),
                "mt_loss_prior": trial_cfg.get("mt_loss_prior", None),
                "regression_label_transform": trial_cfg.get("regression_label_transform", "none"),
                "winsorize_quantile_low": trial_cfg.get("winsorize_quantile_low", 0.01),
                "winsorize_quantile_high": trial_cfg.get("winsorize_quantile_high", 0.99),
                "graph_label_reweight": trial_cfg.get("graph_label_reweight", False),
                "graph_label_bins": trial_cfg.get(
                    "graph_label_bins", [-8.0, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5]
                ),
                "graph_label_weight_power": trial_cfg.get("graph_label_weight_power", 0.5),
                "graph_label_weight_clip": trial_cfg.get("graph_label_weight_clip", 3.0),
                "dist_bar": trial_cfg["dist_bar"],
                "split_pkl_path": trial_cfg.get("split_pkl_path"),
                "allow_empty_test": trial_cfg.get("allow_empty_test", False),
            }
            save_trial_params(
                trial_dir,
                args,
                trial_cfg,
                {"stage1": stage1_trial, "stage2": stage2_trial, "stage3": stage3_trial},
            )
            candidate_model = run_training(args, stage3_trial, "all", stage2_model)

        valid_metric = run_predict_eval_metrics_only(py, project_root, eval_args, pkl_dir, split_root, candidate_model, "valid")
        test_metric = run_predict_eval_metrics_only(py, project_root, eval_args, pkl_dir, split_root, candidate_model, "test")
        final_metric_name = predict_primary_metric_name(valid_metric)
        tqdm.write(
            f"[Trial {idx}/{len(trials)}] {final_tag} best checkpoint re-eval | "
            f"valid_{final_metric_name}={valid_metric.get(final_metric_name):.6f} | "
            f"test_{final_metric_name}={metric_value_text(test_metric, final_metric_name)} | "
            f"model={candidate_model}"
        )
        task_type = valid_metric["task_type"]
        primary_name = predict_primary_metric_name(valid_metric)
        primary_value = valid_metric.get(primary_name)

        candidate = {
            "idx": idx,
            "trial_dir": str(trial_dir),
            "run_tag": final_tag,
            "model": str(candidate_model),
            "stage1_run_tag": stage1_tag,
            "stage2_run_tag": stage2_tag,
            "stage3_run_tag": stage3_tag,
            "stage1_model": str(stage1_model) if stage1_model else None,
            "stage2_model": str(stage2_model) if stage2_model else None,
            "stage3_model": str(candidate_model) if stage3_tag else None,
            "params": {**trial_cfg, "single_stage": args.single_stage},
            "stage1_valid": stage1_valid_metric,
            "stage1_test": stage1_test_metric,
            "stage2_valid": stage2_valid_metric,
            "stage2_test": stage2_test_metric,
            "valid": valid_metric,
            "test": test_metric,
        }
        candidates.append(candidate)
        display_metric_name = primary_name
        display_metric_value = primary_value
        tqdm.write(
            f"[Trial {idx}/{len(trials)}] completed | "
            f"valid_{display_metric_name}={display_metric_value:.6f} | model={candidate_model}"
        )

        if metric_is_better(task_type, primary_value, best_metric):
            best_metric = primary_value
            best_model = candidate_model
            best_candidate = candidate
            tqdm.write(
                f"[Trial {idx}/{len(trials)}] new best | "
                f"valid_{display_metric_name}={display_metric_value:.6f}"
            )

    if best_model is None:
        raise RuntimeError("No successful finetune candidate.")

    final_weights = copy_model_to_result_dir(result_dir, args.task, best_model)
    for candidate in candidates:
        candidate["is_best"] = candidate is best_candidate
        candidate["final_weights"] = str(final_weights) if candidate is best_candidate else None

    final_metrics, score_json = save_scores(py, project_root, eval_settings(args, best_candidate["params"]), final_weights)
    summary_path = result_dir / f"{args.task}_finetune_search_summary.json"
    score_table_csv = result_dir / f"{args.task}_finetune_3step_score_table.csv"
    stage_rows = [row for candidate in candidates for row in build_stage_score_rows(candidate)]
    pd.DataFrame(stage_rows).to_csv(
        score_table_csv,
        index=False,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_candidate": best_candidate,
                "final_weights": str(final_weights),
                "final_metrics": final_metrics,
                "score_table_csv": str(score_table_csv),
                "candidates": candidates,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Three-step Finetune Completed ===")
    print(f"Best model: {best_model}")
    print(f"Final weights: {final_weights}")
    print(f"Score JSON: {score_json}")
    print(f"Score table CSV: {score_table_csv}")
    print(f"Search summary: {summary_path}")


if __name__ == "__main__":
    main()
