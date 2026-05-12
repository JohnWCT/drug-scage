import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--target_col", default="logPapp")
    parser.add_argument("--agg", choices=["median", "mean"], default="median")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {args.smiles_col}")
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    df = df.dropna(subset=[args.smiles_col, args.target_col]).copy()
    df["canonical_smiles"] = df[args.smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()

    grouped = df.groupby("canonical_smiles", as_index=False)
    if args.agg == "median":
        target_agg = grouped[args.target_col].median()
    else:
        target_agg = grouped[args.target_col].mean()

    stats = grouped[args.target_col].agg(
        n_measurements="size",
        label_mean="mean",
        label_median="median",
        label_std="std",
        label_min="min",
        label_max="max",
    )

    out = target_agg.rename(columns={"canonical_smiles": args.smiles_col})
    stats = stats.rename(columns={"canonical_smiles": args.smiles_col})
    out = out.merge(stats, on=args.smiles_col, how="left")
    out["label_range"] = out["label_max"] - out["label_min"]

    simple = out[[args.smiles_col, args.target_col]].copy()
    full_stats_path = output_csv.with_name(output_csv.stem + "_stats.csv")
    simple.to_csv(output_csv, index=False)
    out.to_csv(full_stats_path, index=False)

    print(f"Raw rows: {len(df)}")
    print(f"Unique canonical SMILES: {len(simple)}")
    print(f"Saved clean CSV: {output_csv}")
    print(f"Saved stats CSV: {full_stats_path}")
    print("Duplicate/conflict summary:")
    print(out["label_range"].describe())


if __name__ == "__main__":
    main()
