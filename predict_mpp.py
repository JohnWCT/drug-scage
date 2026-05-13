"""
predict_mpp.py
==============
Per-sample prediction table generator for SCAGE Molecular Property Prediction (MPP).

This script extends ``evaluate_mpp.py`` by:
  * Tracking each sample's SMILES string and original row index throughout inference
  * Writing a structured CSV table that maps every sample to its prediction
    and (if available) its ground-truth label
  * Supporting raw-CSV input so you can evaluate or infer on any molecule set
    without pre-processing it into a PKL first


Operating Modes
---------------
Choose with ``--mode``:

  eval  (default)
      Loads the PKL dataset (``--dataroot``) and evaluates a precomputed split
      from ``--splitroot``.  Use ``--split_type`` + ``--split_seed`` +
      ``--eval_split`` to select subset.

  csv-eval
      Reads a CSV file with ``smiles`` and label column(s), then reports metrics.
      By default, all rows are used.  Add ``--apply_split`` to run only one
      split subset (train/valid/test) generated in-memory.

  infer
      Reads a CSV file with ``smiles`` only (no labels) and outputs predictions.
      By default, all rows are used.  Add ``--apply_split`` to infer one subset.

  pkl-infer
      Loads PKL dataset and infers all rows by default.
      Add ``--apply_split`` to infer one generated split subset.


CLI Arguments
-------------
General
  --mode          eval | csv-eval | infer | pkl-infer      (default: eval)
  --task          Task name, e.g. bbbp, bace, esol, tox21  (default: bbbp)
  --gpus          GPU id(s) passed to CUDA_VISIBLE_DEVICES  (default: 0)
  --batch_size    Batch size for inference                  (default: 32)
  --dataloader_num_workers
                  Number of data-loader worker processes    (default: 4)

Split
  --split_type    scaffold | random_scaffold               (default: scaffold)
  --split_seed    Seed for random_scaffold generated split (default: 8)
  --eval_split    train | valid | test                     (default: test)
  --apply_split   Apply generated split to csv-eval / infer / pkl-infer

Data
  --dataroot      Directory that contains <task>.pkl       (default: ./data/mpp/pkl)
  --splitroot     Directory that contains split pkl files  (default: ./data/mpp/split/)
  --csv           Path to input CSV (csv-eval / infer mode)
  --label_cols    Label column(s) for csv-eval mode. If omitted, use task defaults.

Model
  --ckpt          Path to fine-tuned .pth checkpoint.
                  Default: ./weights/mpp/<task>.pth

Output
  --output        Output CSV path                          (default: ./result/<task>.csv)
  --show          Number of rows printed in preview        (default: 20)


Output CSV Columns
------------------
  sample_id            Original index in the dataset or input CSV
  smiles               SMILES string of the molecule
  pred_score_<col>     Raw sigmoid probability (classification) or
                       raw model output (regression), rounded to 6 dp
  pred_label_<col>     Binary prediction [0/1]  — classification only
  pred_value_<col>     Predicted numeric value  — regression only
  ground_truth_<col>   True label               — omitted in infer mode


Usage Examples
--------------
# 1) Evaluate BBBP on precomputed scaffold test split (classic eval):
    python predict_mpp.py \
        --mode eval \
        --task bbbp \
        --dataroot ./data/mpp/pkl \
        --splitroot ./data/mpp/split \
        --split_type scaffold \
        --eval_split test \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/bbbp_eval_scaffold_test.csv

# 2) Evaluate BBBP on precomputed random_scaffold split (seed=8):
    python predict_mpp.py \
        --mode eval \
        --task bbbp \
        --dataroot ./data/mpp/pkl \
        --splitroot ./data/mpp/split \
        --split_type random_scaffold \
        --split_seed 8 \
        --eval_split test \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/bbbp_eval_random_test.csv

# 3) CSV evaluation with generated random_scaffold split (test subset only):
    python predict_mpp.py \
        --mode csv-eval \
        --task bbbp \
        --csv ./data/mpp/raw/bbbp.csv \
        --split_type random_scaffold \
        --split_seed 8 \
        --apply_split \
        --eval_split test \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/bbbp_csv_eval_random_test.csv

# 4) CSV inference (no labels), only infer scaffold valid subset:
    python predict_mpp.py \
        --mode infer \
        --task bbbp \
        --csv ./my_molecules.csv \
        --split_type scaffold \
        --apply_split \
        --eval_split valid \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/my_molecules_scaffold_valid_predictions.csv

# 5) PKL inference on full dataset (no split):
    python predict_mpp.py \
        --mode pkl-infer \
        --task bbbp \
        --dataroot ./data/mpp/pkl \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/bbbp_all_predictions.csv

# 6) PKL inference on generated random_scaffold train split only:
    python predict_mpp.py \
        --mode pkl-infer \
        --task bbbp \
        --dataroot ./data/mpp/pkl \
        --split_type random_scaffold \
        --split_seed 8 \
        --apply_split \
        --eval_split train \
        --ckpt ./weights/mpp/bbbp.pth \
        --gpus 0 \
        --output ./result/bbbp_pkl_random_train_predictions.csv

Notes
-----
* The script must be run from the project root directory (SCAGE-master/).
* In Docker, prefix with:
    docker exec scage bash -c "cd /workspace/SCAGE-master && python predict_mpp.py ..."
* Weights for evaluation can be downloaded from the links in README.md and placed
  under ``./weights/mpp/<task>.pth``.
"""

import argparse
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from echo_logger import print_debug, dumps_json
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from _config import *
from data_process.compound_tools import CompoundKit, mol_to_data_pkl
from data_process.data_collator import collator_finetune_pkl
from data_process.function_group_constant import nfg
from datasets.dataloader import FinetuneDataset as FinetuneDataset_pkl
from models.scage import Scage
from utils.global_var_util import *
from utils.loss_util import bce_loss
from utils.metric_util import compute_reg_metric_with_r2, compute_cls_metric_tensor_with_pr
from utils.public_util import set_seed
from utils.userconfig_util import (
    config_current_user, config_dataset_form, get_dataset_form, drop_last_flag
)

torch.set_printoptions(sci_mode=False, precision=4, linewidth=400, threshold=1_000_000_000)


# ─────────────────────────────────────────────────────────────────
# Dataset helpers for CSV inputs
# ─────────────────────────────────────────────────────────────────

def _process_row(args):
    """
    Multiprocessing worker: featurise one molecule from a CSV row.

    Parameters
    ----------
    args : tuple
        (original_csv_row_index: int,
         smiles: str,
         label: float | np.ndarray | None)
        ``label`` is None in inference mode.

    Returns
    -------
    (original_csv_row_index, data_dict) or None if SMILES is invalid.
    ``data_dict`` contains graph features produced by ``mol_to_data_pkl`` plus
    the original ``smiles`` string and a ``label`` array (0.0 dummy in infer mode).
    """
    idx, smiles, label = args
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = mol_to_data_pkl(mol)
    data['smiles'] = smiles
    if label is not None:
        data['label'] = np.array([label], dtype=np.float32)
    else:
        # inference mode: set a dummy label = 0 so the collator is happy
        data['label'] = np.array([0.0], dtype=np.float32)
    return (idx, data)


class CsvDataset(Dataset):
    """
    In-memory PyTorch Dataset built on-the-fly from a raw CSV file.

    The CSV must contain a column named ``smiles``.  Optionally, it may also
    contain one or more label columns.  Featurisation is parallelised via
    ``ProcessPoolExecutor``.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    target_cols : list[str] or None
        Column name(s) to use as labels.  Pass ``None`` for inference mode —
        a dummy zero label will be inserted so the collator stays compatible.
    num_workers : int
        Number of parallel worker processes for featurisation.

    Attributes
    ----------
    data : list[dict]
        List of featurised molecule dicts (same format as the PKL datasets).
    original_indices : list[int]
        CSV row index of each successfully featurised molecule.
    smiles_list : list[str]
        SMILES string of each successfully featurised molecule.
    """

    def __init__(self, csv_path: str, target_cols: Optional[List[str]] = None,
                 num_workers: int = 4):
        df = pd.read_csv(csv_path)
        assert 'smiles' in df.columns, "CSV must contain a 'smiles' column."

        has_labels = (target_cols is not None) and all(c in df.columns for c in target_cols)

        args_list = []
        self.original_indices: List[int] = []     # original CSV row index
        self.smiles_list:      List[str]  = []

        for i, row in df.iterrows():
            smi = row['smiles']
            if has_labels:
                # multi-task: average over target columns (same as prepare_data treatment)
                lbl_vals = row[target_cols].values.astype(np.float32)
                lbl = lbl_vals[0] if len(lbl_vals) == 1 else lbl_vals
            else:
                lbl = None
            args_list.append((i, smi, lbl))

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(ex.map(_process_row, args_list),
                            total=len(args_list), desc='Featurising SMILES'):
                results.append(res)

        self.data: List[dict] = []
        self.original_indices = []
        self.smiles_list = []
        for res in results:
            if res is None:
                continue
            orig_idx, d = res
            self.original_indices.append(orig_idx)
            self.smiles_list.append(d['smiles'])
            self.data.append(d)

        print(f"[CsvDataset] {len(self.data)} / {len(df)} molecules successfully featurised.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────

def load_model(config: dict) -> nn.Module:
    """
    Instantiate a SCAGE model and load fine-tuned weights from a checkpoint.

    The function handles both plain state-dicts and the ``{'model': state_dict}``
    wrapper format produced by the training scripts.  DataParallel ``module.``
    prefixes are stripped automatically.

    Parameters
    ----------
    config : dict
        Merged runtime config dict.  Must contain keys:
        ``mode``, ``model``, ``num_tasks``, ``checkpoint``.

    Returns
    -------
    nn.Module
        Model on CUDA in evaluation mode.
    """
    model = Scage(
        mode=config['mode'],
        atom_names=CompoundKit.atom_vocab_dict.keys(),
        atom_embed_dim=config['model']['atom_embed_dim'],
        num_kernel=config['model']['num_kernel'],
        layer_num=config['model']['layer_num'],
        num_heads=config['model']['num_heads'],
        atom_FG_class=nfg() + 1,
        hidden_size=config['model']['hidden_size'],
        num_tasks=config['num_tasks'],
    ).cuda()

    ckpt_path = config['checkpoint']
    print(f'Loading model from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cuda')
    state = ckpt.get('model', ckpt)   # support both formats

    new_dict = model.state_dict()
    # strip DataParallel prefix if present
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k in new_dict}
    new_dict.update(state)
    model.load_state_dict(new_dict)
    model.eval()
    return model


def load_target_transform(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        return ckpt.get('target_transform') or {'type': 'none'}
    return {'type': 'none'}


def inverse_regression_target_transform(values: np.ndarray, transform: dict) -> np.ndarray:
    if not transform or transform.get('type') in [None, 'none', 'None']:
        return values
    if transform.get('type') not in {'standardize', 'winsorized_standardize'}:
        raise ValueError(f"Unsupported target_transform: {transform}")
    mean = np.asarray(transform['mean'], dtype=np.float32).reshape(1, -1)
    std = np.asarray(transform['std'], dtype=np.float32).reshape(1, -1)
    return values * std + mean


def restore_global_dist_bar_from_ckpt(ckpt_path: str):
    """
    Keep collator behavior aligned with evaluate_mpp.py.

    evaluate_mpp.py restores GlobalVar.dist_bar from checkpoint['model']['dist_bar']
    in Trainer.load_ckpt(). The data collator reads GlobalVar.dist_bar at batch time,
    so predict_mpp.py must do the same to get identical metrics.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and ('dist_bar' in state):
        dist_bar = state['dist_bar']
        if torch.is_tensor(dist_bar):
            dist_bar = dist_bar.detach().cpu().numpy().tolist()
        elif hasattr(dist_bar, 'tolist'):
            dist_bar = dist_bar.tolist()
        GlobalVar.dist_bar = list(dist_bar)
        return

    metadata = ckpt.get('metadata', {}) if isinstance(ckpt, dict) else {}
    dist_bar = metadata.get('dist_bar') if isinstance(metadata, dict) else None
    if dist_bar is not None:
        GlobalVar.dist_bar = list(dist_bar)


def load_precomputed_subset(
        dataset: FinetuneDataset_pkl,
        splitroot: str,
        split_type: str,
        task_name: str,
        seed: int,
        split_pkl_path: str = None,
        split_part: str = 'test') -> Subset:
    """Load a precomputed split index file and return the target subset."""
    if split_pkl_path:
        split_path = Path(split_pkl_path)
    elif split_type == 'scaffold':
        split_path = Path(splitroot) / 'scaffold' / f'{task_name}.pkl'
    elif split_type == 'random_scaffold':
        # Flat layout (aligned with 2_data_split.py):
        #   split/random_scaffold/<task>_<seed>.pkl
        split_path = Path(splitroot) / 'random_scaffold' / f'{task_name}_{seed}.pkl'
    elif split_type == 'cyclic_scaffold':
        split_path = Path(splitroot) / 'cyclic_scaffold' / f'{task_name}_cyclic_scaffold_seed{seed}_v0.10_t0.10.pkl'
    else:
        raise ValueError(f'Unsupported split_type: {split_type}')

    if not split_path.exists():
        raise FileNotFoundError(f'Split file not found: {split_path}')

    with open(split_path, 'rb') as f:
        split_idx = pickle.load(f)

    key = f'{split_part}_idx'
    if key not in split_idx:
        raise KeyError(f'Missing key "{key}" in split file: {split_path}')
    return Subset(dataset, split_idx[key])


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=include_chirality,
    )


def build_generated_split_indices(
        smiles_list: List[str],
        split_type: str,
        seed: int,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1) -> Dict[str, List[int]]:
    """
    Build scaffold/random_scaffold split indices in-memory.

    scaffold:
      deterministic ordering (same style as prepare_data.py ScaffoldSplitter)
    random_scaffold:
      pure random 8:1:1 split by sample index (aligned with 2_data_split.py)
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    n_total = len(smiles_list)

    if split_type == 'scaffold':
        all_scaffolds: Dict[str, List[int]] = {}
        for idx, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=False)
            all_scaffolds.setdefault(scaffold, []).append(idx)
        all_scaffolds = {k: sorted(v) for k, v in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for _, scaffold_set in sorted(
                all_scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True
            )
        ]
    elif split_type == 'random_scaffold':
        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(n_total).tolist()
        n_train = int(frac_train * n_total)
        n_valid = int(frac_valid * n_total)
        train_idx = all_indices[:n_train]
        valid_idx = all_indices[n_train:n_train + n_valid]
        test_idx = all_indices[n_train + n_valid:]
        return {
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx,
        }
    else:
        raise ValueError(f'Unsupported split_type for generated split: {split_type}')

    train_cutoff = frac_train * n_total
    valid_cutoff = (frac_train + frac_valid) * n_total
    train_idx, valid_idx, test_idx = [], [], []

    # Match prepare_data.py ScaffoldSplitter behavior.
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx,
    }


# ─────────────────────────────────────────────────────────────────
# Inference loop — returns (smiles_list, preds, labels_or_None)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  smiles_source: List[str],
                  has_labels: bool, task_type: str,
                  return_raw: bool = False):
    """
    Run batch inference over the entire DataLoader without gradient tracking.

    For classification tasks the raw logits are converted to probabilities via
    sigmoid.  For regression tasks the raw model output is returned directly.

    Parameters
    ----------
    model : nn.Module
        SCAGE model in eval mode on CUDA.
    loader : DataLoader
        DataLoader over the evaluation/inference dataset.
    smiles_source : list[str]
        Pre-collected list of SMILES strings in the same order as ``loader``.
        Returned unchanged so the caller can zip them with predictions.
    has_labels : bool
        Whether to collect ground-truth labels from the batch dict.
    task_type : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    smiles_source : list[str]
        Passed through unchanged.
    preds : np.ndarray, shape (N, num_tasks)
        Predicted probabilities (classification) or values (regression).
    labels : np.ndarray or None
        Ground-truth labels if ``has_labels`` is True, else None.
    """
    if len(loader.dataset) == 0:
        empty_preds = np.empty((0, 0), dtype=float)
        empty_labels = np.empty((0, 0), dtype=float) if has_labels else None
        if return_raw:
            return smiles_source, empty_preds, empty_labels, empty_preds
        return smiles_source, empty_preds, empty_labels

    all_preds:  List[np.ndarray] = []
    all_raw:    List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(loader, desc='Inference'):
        batch = {k: v.cuda() for k, v in batch.items()
                 if v is not None and not isinstance(v, list)}
        batch['edge_weight'] = None

        out = model(batch)
        pred = out['graph_feature']  # (bsz, num_tasks)
        all_raw.append(pred.detach().cpu().numpy().copy())

        if task_type == 'classification':
            score = torch.sigmoid(pred)          # probability
        else:
            score = pred                          # raw regression value

        all_preds.append(score.detach().cpu().numpy().copy())
        if has_labels:
            all_labels.append(batch['label'].detach().cpu().numpy().copy())

    if len(all_preds) == 0:
        raise ValueError(
            "No samples to run inference on (empty subset). "
            "Try a different --eval_split or --split_seed, "
            "or disable --apply_split to use all rows."
        )
    preds_arr = np.concatenate(all_preds, axis=0)
    raw_arr = np.concatenate(all_raw, axis=0)
    preds = preds_arr.reshape(len(preds_arr), -1)
    raw_preds = raw_arr.reshape(len(raw_arr), -1)
    labels = None
    if has_labels:
        label_arr = np.concatenate(all_labels, axis=0)
        labels = label_arr.reshape(len(label_arr), -1)

    if return_raw:
        return smiles_source, preds, labels, raw_preds
    return smiles_source, preds, labels


# ─────────────────────────────────────────────────────────────────
# Build output DataFrame + metrics
# ─────────────────────────────────────────────────────────────────

def build_table(mode: str, task_type: str, task_name: str,
                sample_ids: List, smiles_list: List[str],
                preds: np.ndarray, labels,
                target_cols: List[str]) -> pd.DataFrame:
    """
    Assemble the per-sample result DataFrame.

    Columns produced
    ----------------
    sample_id            — original dataset index or CSV row number
    smiles               — molecule SMILES string
    pred_score_<col>     — sigmoid probability (cls) or raw value (reg)
    pred_label_<col>     — binary 0/1 prediction  [classification only]
    pred_value_<col>     — predicted numeric value [regression only]
    ground_truth_<col>   — true label             [omitted in infer mode]

    Parameters
    ----------
    mode        : str          — 'eval', 'csv-eval', or 'infer'
    task_type   : str          — 'classification' or 'regression'
    task_name   : str          — e.g. 'bbbp'
    sample_ids  : list         — original dataset or CSV indices
    smiles_list : list[str]    — SMILES in the same order as predictions
    preds       : np.ndarray   — shape (N, T), model output
    labels      : np.ndarray or None — shape (N, T) or None
    target_cols : list[str]    — column name for each prediction head

    Returns
    -------
    pd.DataFrame
    """
    n_tasks = preds.shape[1] if preds.ndim == 2 else 1
    preds_2d = preds.reshape(len(preds), -1)

    rows = []
    for i, (sid, smi) in enumerate(zip(sample_ids, smiles_list)):
        row = {'sample_id': sid, 'smiles': smi}
        for t, col in enumerate(target_cols):
            score = float(preds_2d[i, t])
            row[f'pred_score_{col}'] = round(score, 6)
            if task_type == 'classification':
                row[f'pred_label_{col}'] = int(score >= 0.5)
            else:
                row[f'pred_value_{col}'] = round(score, 6)
            if labels is not None:
                lbl = float(labels[i].flat[t]) if labels.ndim > 1 else float(labels[i])
                row[f'ground_truth_{col}'] = lbl
        rows.append(row)

    return pd.DataFrame(rows)


def compute_and_print_metrics(task_type: str, task_name: str,
                               preds: np.ndarray, labels: np.ndarray,
                               target_cols: List[str],
                               raw_preds: Optional[np.ndarray] = None,
                               split_name: Optional[str] = None):
    """
    Compute and print evaluation metrics for each prediction head.

    Metric used per task type
    -------------------------
    classification    — ROC-AUC (sklearn).  Samples with label == -1 (missing)
                        are excluded from the calculation.
    regression        — RMSE for most tasks; MAE for QM7/8/9.

    Parameters
    ----------
    task_type   : str         — 'classification' or 'regression'
    task_name   : str         — used to select MAE vs RMSE for QM tasks
    preds       : np.ndarray  — shape (N, T), predicted scores / values
    labels      : np.ndarray  — shape (N, T) or (N,), ground-truth labels
    target_cols : list[str]   — column name for each prediction head
    """
    print("\n" + "=" * 60)
    split_suffix = f" ({split_name})" if split_name else ""
    print(f"  Evaluation Metrics — {task_name}{split_suffix}")
    print("=" * 60)
    preds_2d = preds.reshape(len(preds), -1)
    labels_2d = labels.reshape(len(labels), -1)
    y_true = torch.tensor(labels_2d, device='cuda')
    y_pred_metric = torch.tensor(
        (raw_preds if (task_type == 'classification' and raw_preds is not None) else preds_2d),
        device='cuda'
    )

    if task_type == 'classification':
        roc_auc, pr_auc = compute_cls_metric_tensor_with_pr(y_true, y_pred_metric)
        print(f"  test_auc: {roc_auc:.6f}")
        print(f"  test_pr_auc: {pr_auc:.6f}")
    else:
        mae, rmse, r2 = compute_reg_metric_with_r2(y_true, y_pred_metric)
        print(f"  test_mae: {mae:.6f}")
        print(f"  test_rmse: {rmse:.6f}")
        print(f"  test_r2: {r2:.6f}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    """
    Parse and return command-line arguments.

    Returns
    -------
    argparse.Namespace with fields:
        mode, task, dataroot, splitroot, csv, label_cols,
        ckpt, gpus, batch_size, dataloader_num_workers, output, show
    """
    parser = argparse.ArgumentParser(
        description='SCAGE MPP Predictor — outputs per-sample prediction table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['eval', 'csv-eval', 'infer', 'pkl-infer'],
                        default='eval',
                        help='Operating mode (default: eval)')
    # --- eval mode ---
    parser.add_argument('--task',      type=str, default='bbbp',
                        help='Task name for eval mode (default: bbbp)')
    parser.add_argument('--dataroot',  type=str, default='./data/mpp/pkl',
                        help='PKL dataset directory (eval mode)')
    parser.add_argument('--splitroot', type=str, default='./data/mpp/split/',
                        help='Split directory (eval mode)')
    parser.add_argument('--split_type', choices=['scaffold', 'random_scaffold', 'cyclic_scaffold'],
                        default='scaffold', help='Split strategy for eval mode')
    parser.add_argument('--split_seed', type=int, default=8,
                        help='Seed for random_scaffold generated split (default: 8)')
    parser.add_argument('--split_pkl_path', type=str, default=None,
                        help='Optional explicit split pickle path for eval mode')
    parser.add_argument('--apply_split', action='store_true',
                        help='Apply generated split to pkl-infer/csv-eval/infer modes')
    parser.add_argument('--eval_split', choices=['train', 'valid', 'test'],
                        default='test',
                        help='Which split subset to use (eval mode or --apply_split modes)')
    # --- csv modes ---
    parser.add_argument('--csv',       type=str, default=None,
                        help='Input CSV path (csv-eval / infer mode)')
    parser.add_argument('--label_cols', nargs='+', default=None,
                        help='Column name(s) to use as labels in csv-eval mode. '
                             'If omitted, auto-detect from task config.')
    # --- model ---
    parser.add_argument('--ckpt',      type=str, default=None,
                        help='Checkpoint .pth path. '
                             'Defaults to ./weights/mpp/<task>.pth')
    parser.add_argument('--gpus',      type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    # --- output ---
    parser.add_argument('--output',   type=str, default=None,
                        help='Output CSV path (default: ./result/<task>.csv)')
    parser.add_argument('--show',     type=int, default=20,
                        help='Number of rows to print when --show_table is enabled (default: 20)')
    parser.add_argument('--show_table', action='store_true',
                        help='Print the prediction CSV path and preview rows.')
    parser.add_argument('--quiet_table', action='store_true',
                        help='Deprecated compatibility flag; table output is quiet by default.')
    return parser.parse_args()


def main(base_config: dict):
    """
    Entry point: parse args, build DataLoader, run inference, save results.

    Workflow
    --------
    1. Merge CLI arguments into ``base_config`` (same pattern as finetune_mpp.py).
    2. Build the appropriate DataLoader for the selected mode.
    3. Load the SCAGE model from the specified checkpoint.
    4. Run ``run_inference`` to get per-sample predictions.
    5. Call ``build_table`` to create the result DataFrame.
    6. Save the DataFrame to CSV and print a console preview.
    7. If labels are available, compute and print AUC / RMSE.

    Parameters
    ----------
    base_config : dict
        Config loaded from ``config/config_finetune.yaml``.
    """
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    set_seed(base_config.get('seed', 8))

    # ── Resolve checkpoint ──────────────────────────────────────
    ckpt_path = args.ckpt or f'./weights/mpp/{args.task}.pth'

    # ── Configure base config ───────────────────────────────────
    user = 'mpp'
    base_config['userconfig'][user]['dataset_dir'] = args.dataroot
    base_config['userconfig'][user]['split_dir']   = args.splitroot
    base_config = config_current_user(user, base_config)
    base_config = config_dataset_form('pkl', base_config)
    base_config['task_name']   = args.task
    base_config['split_type']  = args.split_type
    base_config['seed']        = args.split_seed
    base_config['batch_size']  = args.batch_size
    base_config['dataloader_num_workers'] = args.dataloader_num_workers
    base_config = get_downstream_task_names(base_config)

    GlobalVar.dist_bar      = [0, 0]
    base_config['checkpoint'] = ckpt_path
    GlobalVar.freeze_layers  = 0
    base_config['fg_num_']   = nfg() + 1
    base_config['freeze_layers'] = 0
    GlobalVar.parallel_train  = False

    task_type   = base_config['task']        # 'classification' or 'regression'
    target_cols = base_config.get('target', [args.task])
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    # Default output: ./result/<task>.csv  (folder is created automatically)
    DEFAULT_OUTPUT_DIR = './result'
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f'{args.task}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Keep runtime output concise: do not print full config JSON by default.

    # ── Build DataLoader ────────────────────────────────────────
    if args.mode == 'eval':
        # Directly read precomputed split indices and evaluate one subset.
        dataset  = FinetuneDataset_pkl(root=base_config['root'],
                                       task_name=base_config['task_name'])
        eval_dataset = load_precomputed_subset(
            dataset=dataset,
            splitroot=args.splitroot,
            split_type=base_config['split_type'],
            task_name=base_config['task_name'],
            seed=base_config['seed'],
            split_pkl_path=args.split_pkl_path,
            split_part=args.eval_split
        )

        smiles_list  = [dataset.data[i]['smiles'] for i in eval_dataset.indices]
        sample_ids   = list(eval_dataset.indices)
        has_labels   = True
        loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collator_finetune_pkl,
            drop_last=False,
        )

    elif args.mode == 'pkl-infer':
        # Infer on the full PKL dataset (no train/val/test split)
        dataset = FinetuneDataset_pkl(root=base_config['root'],
                                      task_name=base_config['task_name'])
        # Docker containers often have limited /dev/shm. Multi-worker DataLoader
        # can trigger "bus error / No space left on device" when sharing tensors.
        # Force single-process loading in pkl-infer mode for robustness.
        num_workers = 0
        pin_memory = False
        if args.apply_split:
            all_smiles = [one['smiles'] for one in dataset.data]
            generated = build_generated_split_indices(
                smiles_list=all_smiles,
                split_type=args.split_type,
                seed=args.split_seed,
            )
            split_key = f'{args.eval_split}_idx'
            indices = generated[split_key]
            infer_dataset = Subset(dataset, indices)
            sample_ids = list(indices)
            smiles_list = [dataset.data[i]['smiles'] for i in indices]
            has_labels = all(('label' in dataset.data[i]) for i in indices)
        else:
            infer_dataset = dataset
            sample_ids = list(range(len(dataset)))
            smiles_list = [dataset.data[i]['smiles'] for i in sample_ids]
            has_labels = all(('label' in dataset.data[i]) for i in sample_ids)
        loader = DataLoader(
            infer_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator_finetune_pkl,
            # In inference, never drop the tail batch; otherwise small split
            # subsets (len < batch_size) become empty and crash downstream.
            drop_last=False,
        )

    elif args.mode in ('csv-eval', 'infer'):
        assert args.csv, "--csv must be specified in csv-eval / infer mode"
        label_cols  = args.label_cols or (target_cols if args.mode == 'csv-eval' else None)
        has_labels  = (args.mode == 'csv-eval')

        csv_dataset = CsvDataset(
            csv_path=args.csv,
            target_cols=label_cols if has_labels else None,
            num_workers=args.dataloader_num_workers,
        )
        if args.apply_split:
            generated = build_generated_split_indices(
                smiles_list=csv_dataset.smiles_list,
                split_type=args.split_type,
                seed=args.split_seed,
            )
            split_key = f'{args.eval_split}_idx'
            indices = generated[split_key]
            eval_csv_dataset = Subset(csv_dataset, indices)
            smiles_list = [csv_dataset.smiles_list[i] for i in indices]
            sample_ids = [csv_dataset.original_indices[i] for i in indices]
        else:
            eval_csv_dataset = csv_dataset
            smiles_list = csv_dataset.smiles_list
            sample_ids = csv_dataset.original_indices
        loader = DataLoader(
            eval_csv_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,            # already in-memory
            pin_memory=True,
            collate_fn=collator_finetune_pkl,
            drop_last=False,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ── Load model & run ────────────────────────────────────────
    GlobalVar.use_ckpt = True
    model = load_model(base_config)
    restore_global_dist_bar_from_ckpt(ckpt_path)

    smiles_out, preds, labels, raw_preds = run_inference(
        model, loader, smiles_list, has_labels, task_type
        , return_raw=True
    )
    if task_type == 'regression':
        target_transform = load_target_transform(ckpt_path)
        preds = inverse_regression_target_transform(preds, target_transform)
        raw_preds = inverse_regression_target_transform(raw_preds, target_transform)

    # ── Build table ─────────────────────────────────────────────
    df = build_table(
        mode=args.mode, task_type=task_type, task_name=args.task,
        sample_ids=sample_ids, smiles_list=smiles_out,
        preds=preds, labels=labels,
        target_cols=target_cols,
    )

    df.to_csv(output_path, index=False)
    if args.show_table and not args.quiet_table:
        print(f"\nPrediction table saved → {output_path}  ({len(df)} rows)\n")

        # ── Console preview ─────────────────────────────────────────
        pd.set_option('display.max_rows', args.show)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 180)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df.head(args.show).to_string(index=False))

    # ── Metrics ──────────────────────────────────────────────────
    if has_labels and labels is not None:
        metric_split = args.eval_split if args.mode == 'eval' else args.mode
        compute_and_print_metrics(
            task_type,
            args.task,
            preds,
            labels,
            target_cols,
            raw_preds=raw_preds,
            split_name=metric_split,
        )


if __name__ == '__main__':
    path = Path(pdir) / 'config' / 'config_finetune.yaml'
    cfg  = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    main(cfg)
