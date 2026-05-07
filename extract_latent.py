"""
Extract SCAGE latent representations from CSV input.

Supports loading checkpoints from:
  - ./weights/pretrain/*.pth
  - ./weights/mpp/*.pth

Default outputs include graph_token / atom_tokens / atom_mean.

Usage examples
--------------
1) Extract from MPP checkpoint (save graph_token + atom_mean):
    python extract_latent.py \
        --weights ./weights/mpp/bace.pth \
        --task bace \
        --smiles_csv ./data/mpp/raw/bace.csv \
        --smiles_col smiles \
        --id_col drug_name \
        --output ./result/bace_latent.pkl

2) Extract from pretrain checkpoint:
    python extract_latent.py \
        --weights ./weights/pretrain/pretrain.pth \
        --weight_type pretrain \
        --task bace \
        --smiles_csv ./data/mpp/custom/GDSC_drug_merge_pubchem_dropNA_MACCS.csv \
        --smiles_col smiles \
        --id_col drug_name \
        --output ./result/gdsc_pretrain_latent.pkl

3) If CSV has no drug_name column, keep row index as ID:
    python extract_latent.py \
        --weights ./weights/mpp/bace.pth \
        --task bace \
        --smiles_csv ./data/mpp/raw/bace.csv \
        --smiles_col smiles \
        --id_col drug_name \
        --output ./result/bace_latent.pkl
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _config import get_downstream_task_names, model_is_dp, pdir, task_configs
from data_process.compound_tools import CompoundKit, mol_to_data_pkl
from data_process.data_collator import collator_finetune_pkl
from data_process.function_group_constant import nfg
from models.scage import Scage
from utils.global_var_util import GlobalVar


class SmilesFeatureDataset(Dataset):
    def __init__(self, rows: List[Dict], num_tasks: int):
        self.data = []
        self.valid_rows: List[Dict] = []
        self.failed: List[Dict] = []
        for row in rows:
            smi = row["smiles"]
            if smi is None or str(smi).strip() == "" or str(smi).lower() == "nan":
                self.failed.append({**row, "reason": "empty_smiles"})
                continue
            mol = AllChem.MolFromSmiles(smi)
            if mol is None:
                self.failed.append({**row, "reason": "rdkit_parse_failed"})
                continue
            item = mol_to_data_pkl(mol)
            if item is None:
                self.failed.append({**row, "reason": "feature_build_failed"})
                continue
            item["smiles"] = smi
            item["label"] = np.zeros((num_tasks,), dtype=np.float32)
            self.data.append(item)
            self.valid_rows.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SCAGE latent representations from CSV",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract_latent.py --weights ./weights/mpp/bace.pth --task bace "
            "--smiles_csv ./data/mpp/raw/bace.csv --smiles_col smiles --id_col drug_name "
            "--output ./result/bace_latent.pkl\n\n"
            "  python extract_latent.py --weights ./weights/pretrain/pretrain.pth --weight_type pretrain "
            "--smiles_csv ./data/mpp/custom/GDSC_drug_merge_pubchem_dropNA_MACCS.csv "
            "--smiles_col smiles --id_col drug_name "
            "--output ./result/gdsc_pretrain_latent.pkl"
        ),
    )
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path (.pth)")
    parser.add_argument(
        "--weight_type",
        type=str,
        choices=["auto", "pretrain", "mpp"],
        default="auto",
        help="auto: infer from weight path/name (contains pretrain -> pretrain, else mpp)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name. If omitted for mpp, infer from weight filename (e.g., bace.pth -> bace).",
    )
    parser.add_argument("--smiles_csv", type=str, required=True, help="CSV containing a smiles column")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name in CSV")
    parser.add_argument("--id_col", type=str, default="drug_name", help="Preferred ID column name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="./result/latent_output.pkl")
    parser.add_argument(
        "--failed_log_csv",
        type=str,
        default=None,
        help="Optional path to save failed rows (id,row_id,drug_name,smiles,reason).",
    )
    parser.add_argument(
        "--save_atom_tokens",
        action="store_true",
        help="Deprecated; atom tokens are now always exported.",
    )
    return parser.parse_args()


def infer_weight_type(weight_path: str, explicit_type: str) -> str:
    if explicit_type != "auto":
        return explicit_type
    p = weight_path.lower()
    if "/weights/pretrain/" in p or "pretrain" in os.path.basename(p):
        return "pretrain"
    return "mpp"


def resolve_task_name(weight_type: str, weight_path: str, task_arg: Optional[str]) -> str:
    if task_arg:
        return task_arg
    if weight_type == "mpp":
        stem = Path(weight_path).stem
        if stem in task_configs:
            return stem
        raise ValueError(
            f"Cannot infer task from weight filename `{stem}`. "
            f"Please pass --task explicitly (supported: {list(task_configs.keys())})."
        )
    return "bbbp"


def _to_optional_str(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def resolve_rows_from_csv(args) -> List[Dict]:
    df = pd.read_csv(args.smiles_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column `{args.smiles_col}` not found in {args.smiles_csv}")

    has_id_col = args.id_col in df.columns
    rows: List[Dict] = []
    id_counter: Dict[str, int] = {}

    for row_idx, row in df.iterrows():
        smiles = _to_optional_str(row[args.smiles_col])
        drug_name = _to_optional_str(row[args.id_col]) if has_id_col else None
        base_id = drug_name if drug_name is not None else f"row_{row_idx}"
        seen = id_counter.get(base_id, 0)
        unique_id = base_id if seen == 0 else f"{base_id}__dup{seen}"
        id_counter[base_id] = seen + 1

        rows.append(
            {
                "id": unique_id,
                "row_id": int(row_idx),
                "drug_name": drug_name,
                "smiles": smiles,
            }
        )
    return rows


def build_model(weight_type: str, task_name: str, device: str):
    # Scage.__init__ 會直接把 GlobalVar.dist_bar 轉成 tensor，
    # 若尚未設定會是 None，需先給安全預設值。
    if GlobalVar.dist_bar is None:
        GlobalVar.dist_bar = [0, 0]

    cfg_path = Path(pdir) / "config" / "config_finetune.yaml"
    config = yaml.load(open(cfg_path, "r"), Loader=yaml.FullLoader)
    config["task_name"] = task_name
    config = get_downstream_task_names(config)

    model_mode = "pretrain_bin" if weight_type == "pretrain" else "finetune"
    # Align pretrain architecture with released pretrain checkpoint
    # (weights/pretrain/pretrain.pth): finger + sp + angle + fg.
    if model_mode == "pretrain_bin":
        GlobalVar.pretrain_task = ["finger", "sp", "angle", "fg"]

    model = Scage(
        mode=model_mode,
        atom_names=CompoundKit.atom_vocab_dict.keys(),
        atom_embed_dim=config["model"]["atom_embed_dim"],
        num_kernel=config["model"]["num_kernel"],
        layer_num=config["model"]["layer_num"],
        num_heads=config["model"]["num_heads"],
        atom_FG_class=nfg() + 1,
        hidden_size=config["model"]["hidden_size"],
        num_tasks=config["num_tasks"],
    ).to(device)
    return model, config


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    state_dict = ckpt["model"]
    pretrained_keys = set(state_dict.keys())
    if model_is_dp(pretrained_keys):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    if isinstance(state_dict, dict) and "dist_bar" in state_dict:
        dist_bar = state_dict["dist_bar"]
        if torch.is_tensor(dist_bar):
            dist_bar = dist_bar.detach().cpu().numpy().tolist()
        elif hasattr(dist_bar, "tolist"):
            dist_bar = dist_bar.tolist()
        GlobalVar.dist_bar = list(dist_bar)
    return len(matched), len(model_dict)


def encode_tokens(model: Scage, batched_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    atom = model.atom_feature(batched_data)
    atom_attention_mask = batched_data["atom_attention_mask"]
    for i in range(model.layer_num):
        atom = model.EncoderAtomList[i](
            x=atom,
            attn_mask=atom_attention_mask,
            dist=batched_data["pair_distances"],
            dist_bar=batched_data["atom_dist_bar"],
        )
    return atom


def run_extract(model: Scage, loader: DataLoader, device: str):
    model.eval()
    graph_token_list = []
    atom_mean_list = []
    # Per-sample lists: each batch pads atoms to that batch's max_len, so the
    # padded atom dimension differs across batches and cannot be torch.cat on
    # dim=0 with earlier batches. Store one [n_atoms_i, D] / [n_atoms_i] per molecule.
    atom_tokens_rows: List[np.ndarray] = []
    atom_mask_rows: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latent", unit="batch"):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if value is not None and not isinstance(value, list)
            }
            tokens = encode_tokens(model, batch)  # [B, 1+A, D]
            graph_token = tokens[:, 0, :]  # [B, D]
            atom_tokens = tokens[:, 1:, :]  # [B, A_pad, D]
            atom_mask = batch["atom_mask"][:, 1:]  # [B, A_pad]

            # Mask padded atom tokens to 0 so downstream consumers can safely
            # use token tensors without extra masking for padding positions.
            atom_mask_f = atom_mask.unsqueeze(-1).float()
            atom_tokens = atom_tokens * atom_mask_f
            atom_sum = atom_tokens.sum(dim=1)
            atom_cnt = atom_mask_f.sum(dim=1).clamp(min=1.0)
            atom_mean = atom_sum / atom_cnt

            graph_token_list.append(graph_token.cpu())
            atom_mean_list.append(atom_mean.cpu())

            bsz = graph_token.shape[0]
            emb_dim = atom_tokens.shape[-1]
            for bi in range(bsz):
                n_real = int(atom_mask_f[bi].sum().item())
                if n_real > 0:
                    at_np = atom_tokens[bi, :n_real].detach().cpu().numpy()
                    am_np = atom_mask[bi, :n_real].detach().cpu().numpy()
                else:
                    at_np = np.zeros((0, emb_dim), dtype=np.float32)
                    am_np = np.zeros((0,), dtype=np.float32)
                atom_tokens_rows.append(at_np)
                atom_mask_rows.append(am_np)

    out = {
        "graph_token": torch.cat(graph_token_list, dim=0).numpy(),
        "atom_mean": torch.cat(atom_mean_list, dim=0).numpy(),
        "atom_tokens": atom_tokens_rows,
        "atom_mask": atom_mask_rows,
    }
    return out


def build_records(dataset: SmilesFeatureDataset, latent: Dict[str, np.ndarray]) -> List[Dict]:
    records: List[Dict] = []
    n = latent["graph_token"].shape[0]
    for i in range(n):
        row_meta = dataset.valid_rows[i]
        rec = {
            "id": row_meta["id"],
            "row_id": row_meta["row_id"],
            "drug_name": row_meta["drug_name"],
            "smiles": row_meta["smiles"],
            "graph_token": latent["graph_token"][i],
            "atom_mean": latent["atom_mean"][i],
            "atom_tokens": latent["atom_tokens"][i],
            "atom_mask": latent["atom_mask"][i],
        }
        records.append(rec)
    return records


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_type = infer_weight_type(args.weights, args.weight_type)
    task_name = resolve_task_name(weight_type, args.weights, args.task)
    rows = resolve_rows_from_csv(args)
    model, config = build_model(weight_type, task_name, device)
    matched, total = load_checkpoint(model, args.weights)

    dataset = SmilesFeatureDataset(rows, num_tasks=config["num_tasks"])
    if len(dataset) == 0:
        raise RuntimeError("No valid SMILES after RDKit parsing.")
    if len(dataset.failed) > 0:
        print(f"[WARN] Failed molecules: {len(dataset.failed)} / {len(rows)}")
        preview_n = min(10, len(dataset.failed))
        for i in range(preview_n):
            item = dataset.failed[i]
            print(
                f"[WARN] failed[{i}] id={item.get('id')} row_id={item.get('row_id')} "
                f"reason={item.get('reason')} smiles={item.get('smiles')}"
            )
        if len(dataset.failed) > preview_n:
            print(f"[WARN] ... {len(dataset.failed) - preview_n} more failed rows")

        if args.failed_log_csv:
            failed_df = pd.DataFrame(dataset.failed)
            failed_path = Path(args.failed_log_csv)
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_df.to_csv(failed_path, index=False)
            print(f"[WARN] Failed rows saved: {failed_path}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator_finetune_pkl,
        pin_memory=True,
        drop_last=False,
    )

    latent = run_extract(model, loader, device)
    records = build_records(dataset, latent)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(records, f)

    atom_counts = [at.shape[0] for at in latent["atom_tokens"]]
    emb_d = int(latent["graph_token"].shape[1])
    meta = {
        "checkpoint": args.weights,
        "weight_type": weight_type,
        "task_name": task_name,
        "device": device,
        "matched_params": matched,
        "total_model_params": total,
        "input_smiles_count": len(rows),
        "valid_smiles_count": len(dataset),
        "graph_token_shape": list(latent["graph_token"].shape),
        "atom_mean_shape": list(latent["atom_mean"].shape),
        "atom_tokens_layout": "list_per_sample",
        "atom_tokens_count": len(latent["atom_tokens"]),
        "atom_tokens_embedding_dim": emb_d,
        "atom_tokens_atom_count_min": min(atom_counts) if atom_counts else 0,
        "atom_tokens_atom_count_max": max(atom_counts) if atom_counts else 0,
        "atom_mask_layout": "list_per_sample",
        "atom_mask_count": len(latent["atom_mask"]),
        "records_count": len(records),
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Latent extraction finished.")
    print(f"[INFO] Checkpoint type: {weight_type}")
    print(f"[INFO] Task name: {task_name}")
    print(f"[INFO] Params matched: {matched}/{total}")
    print(f"[INFO] Output saved: {output_path}")
    print(f"[INFO] Metadata saved: {meta_path}")
    print(f"[INFO] Output type: records (list)")
    print(f"[INFO] records count: {len(records)}")
    print(f"[INFO] graph_token shape: {latent['graph_token'].shape}")
    print(f"[INFO] atom_mean shape: {latent['atom_mean'].shape}")
    print(
        f"[INFO] atom_tokens: list of {len(latent['atom_tokens'])} arrays, "
        f"emb_dim={emb_d}, n_atoms min={meta['atom_tokens_atom_count_min']}, "
        f"max={meta['atom_tokens_atom_count_max']}"
    )
    print(f"[INFO] atom_mask: list of {len(latent['atom_mask'])} arrays (same order as atom_tokens)")


if __name__ == "__main__":
    main()
