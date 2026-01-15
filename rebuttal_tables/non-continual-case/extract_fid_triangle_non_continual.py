#!/usr/bin/env python3
"""
Build lower-triangular FID tables per run from W&B histories for non-continual experiments.

For each run name formatted as {dataset}-non-continual, iterate
history rows where `task_id` exists. At each task k, W&B logs eval/0..eval/k which
are the FIDs for all tasks up to k. This script collects those into a per-run table.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


@dataclass
class ParsedName:
    dataset: str


def parse_experiment_name(name: str) -> ParsedName:
    if not name:
        return ParsedName(dataset="unknown")
    parts = name.split("-")
    if len(parts) == 1:
        return ParsedName(dataset=parts[0])
    
    dataset = parts[0]
    return ParsedName(dataset=dataset)


def iter_task_rows(run) -> Iterable[Dict]:
    for item in run.scan_history():
        try:
            row = dict(item)
        except Exception:
            try:
                row = item.to_json()  # type: ignore[attr-defined]
            except Exception:
                continue
        if "eval/task_id" in row and row.get("eval/task_id") is not None:
            yield row


def compute_fid_from_row(row: Dict, task_id: int) -> Dict[int, float]:
    fids = {}
    for key, val in row.items():
        if key.startswith("fid-task") and val is not None:
            m = re.match(r"fid-task(\d+)", key)
            if m:
                task_idx = int(m.group(1))
                try:
                    fids[task_idx] = float(val)
                except (ValueError, TypeError):
                    continue
    if task_idx == 0:
        task_id += 1
    return fids, task_id


def extract_triangle_for_run(run, verbose: bool = False) -> pd.DataFrame:
    parsed = parse_experiment_name(run.name or "")
    fids_by_k = {}
    mx = 0
    task_id = -1
    for row in iter_task_rows(run):
        fids, task_id = compute_fid_from_row(row, task_id)
        fids_by_k[task_id] = fids_by_k.get(task_id, {})
        fids_by_k[task_id].update(fids)
    mx = max(len(fids) for fids in fids_by_k.values()) if fids_by_k else 0
    ds = []
    for task_id, fids in fids_by_k.items():
        row = {"task_id": task_id}
        for k in range(mx):
            row[f"fid-task{k}"] = fids.get(k, math.nan)
        ds.append(row)
    df = pd.DataFrame(ds)
    return df, parsed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create per-run lower-triangular FID tables from W&B histories (non-continual).")
    parser.add_argument(
        "--project-path",
        default="ndrsn0208-georgia-institute-of-technology/diffusion-continual-learning",
        type=str,
        help="W&B path 'entity/project'",
    )
    parser.add_argument("--state", type=str, default="finished", help="Filter by run state")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs to process")
    parser.add_argument("--out-dir", type=str, default="fid_triangles_non_continual", help="Directory to write per-run CSV tables")
    parser.add_argument("--combined-long", type=str, default=None, help="Optional path to write a combined long CSV across runs")
    parser.add_argument("--verbose", default=True, action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    if wandb is None:
        print("wandb is not installed. Please 'pip install wandb pandas' and try again.", file=sys.stderr)
        return 2

    api = wandb.Api()
    filters = None if args.state.lower() == "any" else {"state": args.state}
    runs = api.runs(args.project_path, filters=filters)

    os.makedirs(args.out_dir, exist_ok=True)
    combined_frames: List[pd.DataFrame] = []

    count = 0
    for run in runs:
        if args.max_runs is not None and count >= args.max_runs:
            break
        
        # Filter for non-continual experiments
        if "-non-continual" not in (run.name or ""):
            continue
        
        count += 1
        if args.verbose:
            print(f"Processing run {count}: {run.id} | name='{run.name}' | state={run.state}")
        try:
            df_tri, parsed = extract_triangle_for_run(run, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"[error] Skipping run {run.id} due to: {e}")
            continue

        # Write per-run CSV
        out_name = f"{parsed.dataset}_non_continual.csv"
        out_path = os.path.join(args.out_dir, out_name)
        df_tri.to_csv(out_path, index=False)
        if args.verbose:
            print(f"Wrote triangle ({len(df_tri)} rows) to {out_path}")

        if args.combined_long and not df_tri.empty:
            df_tri_with_meta = df_tri.copy()
            df_tri_with_meta.insert(0, "dataset", parsed.dataset)
            combined_frames.append(df_tri_with_meta)

    if args.combined_long:
        combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame(
            columns=["dataset", "task_id"]
        )
        combined.to_csv(args.combined_long, index=False)
        if args.verbose:
            print(f"Wrote combined long CSV to {args.combined_long} ({len(combined)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
