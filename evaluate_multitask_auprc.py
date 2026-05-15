#!/usr/bin/env python3
"""
讀取單一多目標 prediction CSV，逐 target 計算 AUROC 與 AUPRC（與 utils/metric_util.py 一致：
AUROC 使用 sklearn roc_auc_score；AUPRC 使用 precision_recall_curve + auc(recall, precision)）。
ground_truth < 0（例如 -1.0）視為缺失，不納入計算。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

PRED_PREFIX = "pred_score_"
GT_PREFIX = "ground_truth_"
MACRO_TARGET = "__macro_average__"


def discover_targets(columns: list[str]) -> list[str]:
    targets: list[str] = []
    for c in columns:
        if not c.startswith(PRED_PREFIX):
            continue
        name = c[len(PRED_PREFIX):]
        if f"{GT_PREFIX}{name}" in columns:
            targets.append(name)
    return sorted(targets)


def compute_auprc_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def evaluate_one_target(
    scores: np.ndarray, gts: np.ndarray
) -> tuple[float | None, float | None, int, int, int, int, str]:
    n_total = int(len(gts))
    valid = gts >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, None, n_total, 0, 0, 0, "skipped_no_valid"

    y = gts[valid].astype(np.float64)
    s = scores[valid].astype(np.float64)

    pos = int(np.sum(np.isclose(y, 1.0)))
    neg = int(np.sum(np.isclose(y, 0.0)))
    other = n_valid - pos - neg
    if other > 0:
        return None, None, n_total, n_valid, pos, neg, "skipped_non_binary_labels"

    if pos == 0 or neg == 0:
        return None, None, n_total, n_valid, pos, neg, "skipped_single_class"

    auprc = compute_auprc_pr_curve(y, s)
    auroc = float(roc_auc_score(y, s))
    return auprc, auroc, n_total, n_valid, pos, neg, "ok"


def run(input_path: Path, output_path: Path, include_macro_row: bool) -> None:
    df = pd.read_csv(input_path)
    targets = discover_targets(list(df.columns))
    if not targets:
        print("錯誤：找不到成對的 pred_score_<target> 與 ground_truth_<target> 欄位。", file=sys.stderr)
        sys.exit(1)

    dataset_stem = input_path.stem
    file_basename = input_path.name
    rows: list[dict] = []

    for t in targets:
        pred_col = f"{PRED_PREFIX}{t}"
        gt_col = f"{GT_PREFIX}{t}"
        scores = df[pred_col].to_numpy(dtype=np.float64, copy=False)
        gts = df[gt_col].to_numpy(dtype=np.float64, copy=False)

        auprc, auroc, n_total, n_valid, n_pos, n_neg, status = evaluate_one_target(scores, gts)
        rows.append(
            {
                "dataset": dataset_stem,
                "file": file_basename,
                "target": t,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "auroc": auroc if auroc is not None else np.nan,
                "auprc": auprc if auprc is not None else np.nan,
                "status": status,
            }
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_aurocs = [r["auroc"] for r in ok_rows if not np.isnan(r["auroc"])]
    ok_auprcs = [r["auprc"] for r in ok_rows if not np.isnan(r["auprc"])]
    macro_auroc = float(np.mean(ok_aurocs)) if ok_aurocs else float("nan")
    macro_auprc = float(np.mean(ok_auprcs)) if ok_auprcs else float("nan")
    n_ok = len(ok_auprcs)
    print(f"輸入：{input_path}")
    print(
        f"targets 總數：{len(targets)}，成功計算 AUROC/AUPRC：{n_ok}，"
        f"macro average AUROC：{macro_auroc:.6f}，macro average AUPRC：{macro_auprc:.6f}"
    )

    out_df = pd.DataFrame(rows)
    int_cols = ["n_total", "n_valid", "n_positive", "n_negative"]
    for c in int_cols:
        out_df[c] = out_df[c].astype("Int64")
    if include_macro_row and ok_rows:
        macro_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_stem,
                    "file": file_basename,
                    "target": MACRO_TARGET,
                    "n_total": pd.NA,
                    "n_valid": pd.NA,
                    "n_positive": pd.NA,
                    "n_negative": pd.NA,
                    "auroc": macro_auroc,
                    "auprc": macro_auprc,
                    "status": "macro_average_over_ok_targets",
                }
            ]
        )
        for c in int_cols:
            macro_df[c] = macro_df[c].astype("Int64")
        out_df = pd.concat([out_df, macro_df], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"已寫入：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多目標 prediction CSV 逐 target AUROC 與 AUPRC 總表")
    parser.add_argument(
        "input_csv",
        type=Path,
        help="單一 prediction CSV 路徑（需含 pred_score_<t> 與 ground_truth_<t>）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("result/auprc_summary.csv"),
        help="輸出總表 CSV（預設：result/auprc_summary.csv）",
    )
    parser.add_argument(
        "--no-macro-row",
        action="store_true",
        help="不在輸出 CSV 末尾附加 __macro_average__ 彙總列",
    )
    args = parser.parse_args()
    inp = args.input_csv.resolve()
    if not inp.is_file():
        print(f"錯誤：找不到檔案 {inp}", file=sys.stderr)
        sys.exit(1)
    run(inp, args.output.resolve(), include_macro_row=not args.no_macro_row)


if __name__ == "__main__":
    main()
