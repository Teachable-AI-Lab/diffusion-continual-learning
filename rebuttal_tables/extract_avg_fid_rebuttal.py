#!/usr/bin/env python3
"""
Extract avg_fid across W&B runs for top-eig experiments.

Rules:
- Run names follow: {dataset}-top-eig{num}-{seed}.
  - if the last token isn't an int, it belongs to method and seed defaults to 123.
- For each run, fetch full history (no sampling) and keep rows where `task_id` exists.
- At each such row, prefer `avg_fid` (or similar aliases). If absent, average FID across
  columns `eval/{i}` for i in [0, task_id].

Outputs a CSV in either:
- long: run_name, dataset, num_eigs, seed, _step, task_id, avg_fid
- wide (default): dataset, num_eigs, seed, avg_fid@task0, avg_fid@task1, ...
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
	import wandb  # type: ignore
except Exception as e:  # pragma: no cover - import error reported at runtime
	wandb = None  # Allows --help to work without wandb installed


DEFAULT_SEED = 123
FID_KEY_CANDIDATES = [
	"avg_fid",
	"aver_fid",
	"avg-fid",
	"aver-fid",
	"average_fid",
	"fid_avg",
]


@dataclass
class ParsedName:
	dataset: str
	num_eigs: int
	seed: int


def parse_experiment_name(name: str) -> ParsedName:
	"""Parse `{dataset}-top-eig{num}-{seed}` allowing variations.

	If the last token is not an integer, seed defaults to 123.
	"""
	if not name:
		return ParsedName(dataset="unknown", num_eigs=0, seed=DEFAULT_SEED)

	parts = name.split("-")
	if len(parts) == 1:
		return ParsedName(dataset=parts[0], num_eigs=0, seed=DEFAULT_SEED)

	dataset = parts[0]
	
	# Look for top-eig{num} pattern
	num_eigs = 0
	for part in parts:
		if part.startswith("top"):
			continue
		if part.startswith("eig"):
			try:
				num_eigs = int(part[3:])  # Extract number after 'eig'
				break
			except ValueError:
				pass
	
	# Check if last part is seed
	maybe_seed = parts[-1]
	seed: int
	try:
		seed = int(maybe_seed)
	except ValueError:
		seed = DEFAULT_SEED

	return ParsedName(dataset=dataset, num_eigs=num_eigs, seed=seed)


def pick_avg_fid(row: Dict) -> Optional[float]:
	"""Get avg_fid from common keys if present and not NaN."""
	for key in FID_KEY_CANDIDATES:
		if key in row:
			val = row.get(key)
			if val is not None and not (isinstance(val, float) and math.isnan(val)):
				try:
					return float(val)
				except Exception:
					continue
	return None


_EVAL_COL_RE = re.compile(r"^eval/(\d+)$")


def compute_avg_fid_from_eval_cols(row: Dict, up_to_task_id: int) -> Optional[float]:
	"""Compute mean over eval/{i} for i in [0, up_to_task_id] when explicit avg isn't present."""
	vals: List[float] = []
	for key, val in row.items():
		m = _EVAL_COL_RE.match(str(key))
		if not m:
			continue
		try:
			tid = int(m.group(1))
		except Exception:
			continue
		if tid <= up_to_task_id and val is not None:
			try:
				fv = float(val)
				if not math.isnan(fv) and math.isfinite(fv):
					vals.append(fv)
			except Exception:
				continue
	if not vals:
		return None
	return float(sum(vals) / len(vals))


def iter_task_rows(run) -> Iterable[Dict]:
	"""Yield dict rows from scan_history that contain a non-null task_id."""
	for item in run.scan_history():
		try:
			row = dict(item)
		except Exception:
			try:
				row = item.to_json()  # type: ignore[attr-defined]
			except Exception:
				continue
		if "task_id" in row and row.get("task_id") is not None:
			yield row


def extract_for_run(run, verbose: bool = False) -> List[Dict]:
	"""Extract summary rows for a single W&B run."""
	parsed = parse_experiment_name(run.name or "")
	results: List[Dict] = []

	for row in iter_task_rows(run):
		task_id_raw = row.get("task_id")
		try:
			task_id = int(task_id_raw)
		except Exception:
			if verbose:
				print(f"[skip] run {run.id} row with non-int task_id={task_id_raw}")
			continue

		avg_fid = pick_avg_fid(row)
		if avg_fid is None:
			avg_fid = compute_avg_fid_from_eval_cols(row, task_id)

		step = row.get("_step")

		results.append(
			{
				"run_name": run.name,
				"dataset": parsed.dataset,
				"num_eigs": parsed.num_eigs,
				"seed": parsed.seed,
				"_step": step,
				"task_id": task_id,
				"avg_fid": avg_fid,
			}
		)

	return results


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Extract avg_fid at each task_id from W&B runs (top-eig experiments).")
	parser.add_argument(
		"--project-path",
		default="ndrsn0208-georgia-institute-of-technology/diffusion-continual-learning",
		type=str,
		help="W&B path 'entity/project'",
	)
	parser.add_argument("--state", type=str, default="finished", help="Filter by run state (finished|running|crashed|any)")
	parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs to process")
	parser.add_argument("--output", type=str, default="avg_fid_summary_rebuttal.csv", help="Output CSV file path")
	parser.add_argument(
		"--format",
		type=str,
		choices=["wide", "long"],
		default="wide",
		help="Output format: wide -> one row per (dataset,num_eigs,seed) with avg_fid@taskN columns; long -> one row per task_id.",
	)
	parser.add_argument(
		"--fillna",
		type=float,
		default=None,
		help="Fill missing avg_fid values in wide format with this number (default: leave as blank/NaN).",
	)
	parser.add_argument("--verbose", default=True, action="store_true", help="Verbose logging")

	args = parser.parse_args(argv)

	if wandb is None:
		print("wandb is not installed. Please 'pip install wandb pandas' and try again.", file=sys.stderr)
		return 2

	api = wandb.Api()

	filters = None
	if args.state and args.state.lower() != "any":
		filters = {"state": args.state}

	if args.verbose:
		print(f"Listing runs for {args.project_path} with filters={filters or {}}...")

	runs = api.runs(args.project_path, filters=filters)  # type: ignore[arg-type]

	all_rows: List[Dict] = []
	count = 0
	for run in runs:
		if args.max_runs is not None and count >= args.max_runs:
			break
		
		# Filter for top-eig experiments
		if "top-eig" not in (run.name or ""):
			continue
		
		count += 1

		if args.verbose:
			print(f"Processing run {count}: {run.id} | name='{run.name}' | state={run.state}")

		try:
			rows = extract_for_run(run, verbose=args.verbose)
		except Exception as e:
			if args.verbose:
				print(f"[error] Skipping run {run.id} due to: {e}")
			continue

		all_rows.extend(rows)

	if not all_rows:
		if args.verbose:
			print("No rows found with task_id. Writing empty CSV.")
		if args.format == "wide":
			df_empty = pd.DataFrame(columns=["dataset", "num_eigs", "seed"])
		else:
			df_empty = pd.DataFrame(columns=["run_name", "dataset", "num_eigs", "seed", "_step", "task_id", "avg_fid"])
		df_empty.to_csv(args.output, index=False)
		if args.verbose:
			print(f"Wrote empty CSV to {args.output}")
		return 0

	df_long = pd.DataFrame(all_rows)
	if "task_id" in df_long.columns:
		df_long["task_id"] = pd.to_numeric(df_long["task_id"], errors="coerce")
	if "seed" in df_long.columns:
		df_long["seed"] = pd.to_numeric(df_long["seed"], errors="coerce").astype("Int64")
	if "avg_fid" in df_long.columns:
		df_long["avg_fid"] = pd.to_numeric(df_long["avg_fid"], errors="coerce")
	if "num_eigs" in df_long.columns:
		df_long["num_eigs"] = pd.to_numeric(df_long["num_eigs"], errors="coerce").astype("Int64")

	if args.format == "long":
		sort_cols = [c for c in ["dataset", "num_eigs", "seed", "task_id"] if c in df_long.columns]
		if sort_cols:
			df_long = df_long.sort_values(sort_cols)

		out_path = args.output
		os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
		df_long.to_csv(out_path, index=False)

		if args.verbose:
			with pd.option_context("display.max_columns", None, "display.width", 200):
				print(df_long.head(20))
			print(f"Wrote {len(df_long)} rows to {out_path}")
		return 0

	# Wide format
	dfw = df_long[["dataset", "num_eigs", "seed", "task_id", "avg_fid"]].copy()
	dfw = dfw.dropna(subset=["task_id"])
	dfw = (
		dfw.groupby(["dataset", "num_eigs", "seed", "task_id"], dropna=False, as_index=False)["avg_fid"].mean()
	)
	pivot = dfw.pivot_table(
		index=["dataset", "num_eigs", "seed"],
		columns="task_id",
		values="avg_fid",
		aggfunc="mean",
	)
	pivot = pivot.sort_index(axis=1)
	pivot.columns = [f"avg_fid@task{int(c)}" for c in pivot.columns]
	pivot = pivot.reset_index()

	if args.fillna is not None:
		pivot = pivot.fillna(args.fillna)

	out_path = args.output
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	pivot.to_csv(out_path, index=False)

	if args.verbose:
		with pd.option_context("display.max_columns", None, "display.width", 200):
			print(pivot.head(20))
		print(f"Wrote {len(pivot)} rows to {out_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
