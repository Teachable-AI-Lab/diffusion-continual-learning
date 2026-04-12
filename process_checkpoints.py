#!/usr/bin/env python3
"""Aggregate checkpoint CSVs by model seeds and plot per-task learning curves.

Usage:
  python scratch/process_checkpoints.py --root scratch/icml-checkpoint-tables --out scratch/processed_checkpoints --plot

This will:
  - Search `--root` recursively for CSV files
  - Group model directories that differ only by a trailing `-<seed>` (e.g. `modelname-123`)
  - Compute mean and std across seeds at each `trained_task` for each `task*` column
  - Write per-model per-task CSVs to `--out` and save plots (one plot per target task showing tasks j<=i)
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SEED_SUFFIX = re.compile(r"(.+?)-\d+$")


def find_csvs(root: Path):
    return list(root.rglob("*.csv"))


def base_model_name(name: str) -> str:
    m = SEED_SUFFIX.match(name)
    return m.group(1) if m else name


def load_seed_dfs(csv_paths):
    # Map: base_model -> dict(seed_name -> DataFrame)
    models = {}
    for p in csv_paths:
        parent = p.parent.name
        base = base_model_name(parent)
        seed = parent
        df = pd.read_csv(p)
        if 'trained_task' not in df.columns:
            raise RuntimeError(f"CSV {p} missing 'trained_task' column")
        df = df.set_index('trained_task').sort_index()
        models.setdefault(base, {})[seed] = df
    return models


def aggregate_model(seed_dfs: dict):
    # seed_dfs: seed -> DataFrame (index trained_task, columns task0...)
    # Build union index of steps
    all_steps = sorted({s for df in seed_dfs.values() for s in df.index})
    # collect columns union
    all_cols = sorted({c for df in seed_dfs.values() for c in df.columns})
    arrs = []
    seeds = []
    for seed, df in seed_dfs.items():
        reindexed = df.reindex(all_steps)[all_cols]
        arrs.append(reindexed.values.astype(float))
        seeds.append(seed)
    data = np.stack(arrs, axis=0)  # shape (n_seeds, n_steps, n_tasks)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    mean_df = pd.DataFrame(mean, index=all_steps, columns=all_cols)
    std_df = pd.DataFrame(std, index=all_steps, columns=all_cols)
    return mean_df, std_df


def write_per_task_csvs(out_base: Path, base_model: str, mean_df: pd.DataFrame, std_df: pd.DataFrame):
    model_dir = out_base / base_model
    (model_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (model_dir / 'per_task').mkdir(parents=True, exist_ok=True)
    for col in mean_df.columns:
        out = pd.DataFrame({
            'step': mean_df.index,
            'mean': mean_df[col].values,
            'std': std_df[col].values,
        })
        out.to_csv(model_dir / 'per_task' / f"{col}.csv", index=False)


def plot_for_model(out_base: Path, base_model: str, mean_df: pd.DataFrame, std_df: pd.DataFrame):
    model_dir = out_base / base_model
    cols = mean_df.columns.tolist()
    # assume columns like task0, task1, ... and numeric order
    def task_index(c):
        m = re.match(r"task(\d+)$", c)
        return int(m.group(1)) if m else 1_000_000

    cols = sorted(cols, key=task_index)
    for i, target in enumerate(cols):
        # show tasks j <= i
        to_plot = cols[: i + 1]
        plt.figure(figsize=(8, 5))
        for c in to_plot:
            steps = mean_df.index
            mean = mean_df[c].values
            std = std_df[c].values
            plt.plot(steps, mean, label=c)
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
        plt.xlabel('checkpoint (trained_task)')
        plt.ylabel('metric')
        plt.title(f"{base_model} — learning curves for {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / 'plots' / f"{target}_curve.png")
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/storage/home/hcoda1/1/agupta886/scratch/icml-checkpoint-tables', help='root with model CSV dirs')
    p.add_argument('--out', default='/storage/home/hcoda1/1/agupta886/scratch/processed_checkpoints', help='output folder')
    p.add_argument('--plot', action='store_true', help='generate plots')
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csvs = find_csvs(root)
    if not csvs:
        print('No CSVs found under', root)
        return

    models = load_seed_dfs(csvs)
    print(f'Found {len(models)} base-models')
    for base, seed_map in models.items():
        print('Processing', base, 'with', len(seed_map), 'seeds')
        mean_df, std_df = aggregate_model(seed_map)
        write_per_task_csvs(out, base, mean_df, std_df)
        if args.plot:
            plot_for_model(out, base, mean_df, std_df)

    print('Done — processed outputs in', out)


if __name__ == '__main__':
    main()
