#!/usr/bin/env python3
"""Find rows in CSV files that match an input SMILES by string or structure.

This script is intended to run inside the SCAGE Docker container, for example:

    python find_smiles_in_csv.py --smiles 'COC(=O)C(F)(F)F' --path data/mpp/raw/freesolv.csv
    python find_smiles_in_csv.py --smiles 'COC(=O)C(F)(F)F' --path data/mpp/raw
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rdkit import Chem


COMMON_SMILES_COLUMNS = (
    "smiles",
    "SMILES",
    "Smiles",
    "canonical_smiles",
    "Canonical_SMILES",
    "mol",
    "molecule",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search CSV files for rows matching an input SMILES string or chemical structure."
    )
    parser.add_argument(
        "--smiles",
        required=True,
        help="Input SMILES to search for.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="CSV file or directory containing CSV files.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Column name containing SMILES. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --path is a directory, recursively scan subdirectories.",
    )
    parser.add_argument(
        "--ignore-stereochemistry",
        action="store_true",
        help="Compare structures without stereochemistry.",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str, *, ignore_stereochemistry: bool = False) -> Optional[str]:
    """Convert SMILES to RDKit canonical SMILES; return None if parsing fails."""
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not ignore_stereochemistry)


def iter_csv_files(path: Path, *, recursive: bool = False) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Input path is a file but not a CSV: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path.glob(pattern))


def detect_smiles_column(
    fieldnames: Iterable[str],
    requested_column: Optional[str] = None,
) -> str:
    columns = list(fieldnames)

    if requested_column is not None:
        if requested_column not in columns:
            raise ValueError(f"Requested SMILES column '{requested_column}' not found. Columns: {columns}")
        return requested_column

    for column in COMMON_SMILES_COLUMNS:
        if column in columns:
            return column

    raise ValueError(
        "Could not auto-detect a SMILES column. "
        f"Please pass --smiles-column. Columns: {columns}"
    )


def search_csv(
    csv_path: Path,
    query_smiles: str,
    query_canonical: str,
    *,
    smiles_column: Optional[str],
    ignore_stereochemistry: bool,
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return matches

        detected_column = detect_smiles_column(reader.fieldnames, smiles_column)

        for row_index, row in enumerate(reader, start=1):
            row_smiles = (row.get(detected_column) or "").strip()
            if not row_smiles:
                continue

            exact_match = row_smiles == query_smiles
            row_canonical = canonicalize_smiles(
                row_smiles,
                ignore_stereochemistry=ignore_stereochemistry,
            )
            structure_match = row_canonical == query_canonical

            if exact_match or structure_match:
                matches.append(
                    {
                        "file": str(csv_path),
                        "csv_line": row_index + 1,  # +1 because line 1 is the header
                        "data_row_index_1based": row_index,
                        "smiles_column": detected_column,
                        "row_smiles": row_smiles,
                        "row_canonical_smiles": row_canonical,
                        "exact_smiles_match": exact_match,
                        "structure_match": structure_match,
                        "row": row,
                    }
                )

    return matches


def print_matches(matches: List[Dict[str, object]], query_canonical: str) -> None:
    if not matches:
        print("No matching rows found.")
        print(f"query_canonical_smiles: {query_canonical}")
        return

    print(f"Found {len(matches)} matching row(s).")
    print(f"query_canonical_smiles: {query_canonical}")
    print()

    for idx, match in enumerate(matches, start=1):
        print(f"[Match {idx}]")
        print(f"file: {match['file']}")
        print(f"csv_line: {match['csv_line']}")
        print(f"data_row_index_1based: {match['data_row_index_1based']}")
        print(f"smiles_column: {match['smiles_column']}")
        print(f"row_smiles: {match['row_smiles']}")
        print(f"row_canonical_smiles: {match['row_canonical_smiles']}")
        print(f"exact_smiles_match: {match['exact_smiles_match']}")
        print(f"structure_match: {match['structure_match']}")
        print("row_information:")
        print(json.dumps(match["row"], ensure_ascii=False, indent=2))
        print()


def main() -> None:
    args = parse_args()

    query_canonical = canonicalize_smiles(
        args.smiles,
        ignore_stereochemistry=args.ignore_stereochemistry,
    )
    if query_canonical is None:
        raise ValueError(f"Invalid input SMILES: {args.smiles}")

    csv_files = iter_csv_files(args.path, recursive=args.recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {args.path}")

    all_matches: List[Dict[str, object]] = []
    for csv_file in csv_files:
        all_matches.extend(
            search_csv(
                csv_file,
                args.smiles,
                query_canonical,
                smiles_column=args.smiles_column,
                ignore_stereochemistry=args.ignore_stereochemistry,
            )
        )

    print_matches(all_matches, query_canonical)


if __name__ == "__main__":
    main()
