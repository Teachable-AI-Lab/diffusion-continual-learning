#!/usr/bin/env python3
"""Compute average forgetting from precomputed FID triangle CSVs.

Input directory (default: fid_triangles/) contains per-run CSVs produced by
`extract_fid_triangle.py`. Each file name is expected to be of the form:

	{dataset}_{method}_{seed}.csv

Where method may itself contain underscores (we assume the last token is an
integer seed; if not, seed defaults to 123 and the token is part of the method).

Each CSV has rows for task_id progression (0..K) and columns like:
	task_id,fid-task0,fid-task1,...

Definitions:
  - For task t, the "initial" FID is taken from the row where task_id == t and
	the column fid-task{t}.
  - The "final" FID for task t is taken from the LAST row (max task_id) in the
	same column fid-task{t}.
  - Forgetting for task t = final_fid - initial_fid.
	(Positive means performance (FID) worsened; adjust externally if you define
	 improvement differently.)

Outputs (wide format by default):
	dataset,method,seed,forgetting@task0,forgetting@task1,...,avg_forgetting

Optional long format (--long) with columns:
	dataset,method,seed,task_id,forgetting

Missing Data Handling:
  - If either initial or final FID is NaN/missing, forgetting for that task is NaN.
  - Use --fillna VALUE to fill NaNs in the wide output (avg_forgetting is then
	recomputed after fill if desired; by default avg_forgetting ignores NaNs).

Example usage:
	python fid_triangle_avg_forgetting.py --triangles-dir fid_triangles \
		--output fid_avg_forgetting.csv --long fid_avg_forgetting_long.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_SEED = 123


@dataclass
class ParsedFilename:
	dataset: str
	method: str
	seed: int
	filename: str


_SEED_RE = re.compile(r"^(\d+)$")


def parse_filename(fname: str) -> ParsedFilename:
	"""Parse dataset, method, seed from filename stem dataset_method_seed.csv.

	Heuristic: split stem by '_' ; last token that is int => seed; else seed=DEFAULT_SEED.
	Remaining middle tokens (>=1) joined by '_' are method; first token is dataset.
	If only one token, dataset=token, method="".
	"""
	stem = os.path.splitext(os.path.basename(fname))[0]
	parts = stem.split("_")
	if not parts:
		return ParsedFilename("unknown", "unknown", DEFAULT_SEED, fname)
	if len(parts) == 1:
		return ParsedFilename(parts[0], "", DEFAULT_SEED, fname)
	maybe_seed = parts[-1]
	seed: int
	method_tokens: List[str]
	if _SEED_RE.match(maybe_seed):
		seed = int(maybe_seed)
		method_tokens = parts[1:-1]
	else:
		seed = DEFAULT_SEED
		method_tokens = parts[1:]
	method = "_".join(method_tokens) if method_tokens else ""
	return ParsedFilename(parts[0], method, seed, fname)


def compute_forgetting(df: pd.DataFrame) -> Tuple[pd.Series, List[int]]:
	"""Compute per-task forgetting given a triangle DataFrame.

	Returns (forgetting_series, task_ids_sorted).
	forgetting_series indexed by task_id integer: final - initial.
	"""
	if df.empty or "task_id" not in df.columns:
		return pd.Series(dtype=float), []
	# Ensure task_id numeric
	df = df.copy()
	df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
	df = df.dropna(subset=["task_id"])  # require valid
	if df.empty:
		return pd.Series(dtype=float), []
	df["task_id"] = df["task_id"].astype(int)
	max_task_id = df["task_id"].max()
	last_row = df.loc[df["task_id"].idxmax()]
	forgetting = {}
	task_ids_sorted: List[int] = []
	for t in sorted(df["task_id"].unique()):
		col = f"fid-task{t}"  # expected column name
		if col not in df.columns:
			continue  # can't compute
		# initial fid: row where task_id == t, column fid-task{t}
		init_row = df.loc[df["task_id"] == t]
		if init_row.empty:
			continue
		init_val = init_row.iloc[0].get(col)
		final_val = last_row.get(col)
		try:
			init_f = float(init_val) if init_val is not None else math.nan
		except Exception:
			init_f = math.nan
		try:
			final_f = float(final_val) if final_val is not None else math.nan
		except Exception:
			final_f = math.nan
		if (isinstance(init_f, float) and math.isnan(init_f)) or (isinstance(final_f, float) and math.isnan(final_f)):
			forgetting[t] = math.nan
		else:
			forgetting[t] = final_f - init_f
		task_ids_sorted.append(t)
	return pd.Series(forgetting), task_ids_sorted


def main(argv: Optional[List[str]] = None) -> int:
	p = argparse.ArgumentParser(description="Compute average forgetting from FID triangle CSVs.")
	p.add_argument("--triangles-dir", default="fid_triangles", type=str, help="Directory containing triangle CSVs")
	p.add_argument("--output", default="fid_avg_forgetting.csv", type=str, help="Output wide CSV path")
	p.add_argument("--long", default=None, type=str, help="Optional path to write long format CSV")
	p.add_argument("--fillna", default=None, type=float, help="Fill NaNs in forgetting columns with this value before computing avg_forgetting if provided")
	p.add_argument("--verbose", action="store_true", help="Verbose logging")
	args = p.parse_args(argv)

	in_dir = args.triangles_dir
	if not os.path.isdir(in_dir):
		raise SystemExit(f"Input directory not found: {in_dir}")

	files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.csv')]
	if args.verbose:
		print(f"Found {len(files)} CSV files in {in_dir}")
	rows_wide = []
	long_records = []
	all_task_ids: set[int] = set()

	for fpath in sorted(files):
		parsed = parse_filename(fpath)
		try:
			df = pd.read_csv(fpath)
		except Exception as e:
			if args.verbose:
				print(f"[skip] Could not read {fpath}: {e}")
			continue
		forgetting_series, task_ids = compute_forgetting(df)
		if args.verbose:
			print(f"Processed {os.path.basename(fpath)} tasks={task_ids}")
		all_task_ids.update(task_ids)
		row = {"dataset": parsed.dataset, "method": parsed.method, "seed": parsed.seed}
		for t, val in forgetting_series.items():
			row[f"forgetting@task{t}"] = val
			if args.long is not None:
				long_records.append({
					"dataset": parsed.dataset,
					"method": parsed.method,
					"seed": parsed.seed,
					"task_id": t,
					"forgetting": val,
				})
		rows_wide.append(row)

	if not rows_wide:
		if args.verbose:
			print("No data rows produced; writing empty outputs.")
		# Write empty wide
		empty_cols = ["dataset", "method", "seed"]
		pd.DataFrame(columns=empty_cols).to_csv(args.output, index=False)
		if args.long:
			pd.DataFrame(columns=["dataset", "method", "seed", "task_id", "forgetting"]).to_csv(args.long, index=False)
		return 0

	# Build wide DataFrame
	wide_df = pd.DataFrame(rows_wide)
	# Ensure all forgetting columns exist (even if missing in some rows)
	for t in sorted(all_task_ids):
		col = f"forgetting@task{t}"
		if col not in wide_df.columns:
			wide_df[col] = math.nan

	forgetting_cols = [c for c in wide_df.columns if c.startswith("forgetting@task")]
	# (Optional) fill NaNs
	if args.fillna is not None:
		wide_df[forgetting_cols] = wide_df[forgetting_cols].fillna(args.fillna)
		# avg_forgetting: simple mean of these columns
		wide_df["avg_forgetting"] = wide_df[forgetting_cols].mean(axis=1)
	else:
		# avg_forgetting ignoring NaNs
		wide_df["avg_forgetting"] = wide_df[forgetting_cols].mean(axis=1, skipna=True)

	# Sort for readability
	sort_cols = [c for c in ["dataset", "method", "seed"] if c in wide_df.columns]
	wide_df = wide_df.sort_values(sort_cols)

	os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
	wide_df.to_csv(args.output, index=False)
	if args.verbose:
		with pd.option_context('display.max_columns', None, 'display.width', 180):
			print(wide_df.head(20))
		print(f"Wrote wide forgetting table to {args.output} ({len(wide_df)} rows)")

	if args.long:
		long_df = pd.DataFrame(long_records)
		if not long_df.empty:
			long_df = long_df.sort_values(["dataset", "method", "seed", "task_id"])
		os.makedirs(os.path.dirname(args.long) or '.', exist_ok=True)
		long_df.to_csv(args.long, index=False)
		if args.verbose:
			print(f"Wrote long forgetting table to {args.long} ({len(long_df)} rows)")

	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
