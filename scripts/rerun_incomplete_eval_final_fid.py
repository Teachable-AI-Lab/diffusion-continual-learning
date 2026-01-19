#!/usr/bin/env python3
"""Check FID evaluation outputs and resubmit missing tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_MODELS_ROOT = Path("/storage/home/hcoda1/1/agupta886/scratch/icml-2026")
DEFAULT_SUBMIT_SCRIPT = Path("submit_eval_final_fid_jobs.py")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Identify missing per-task FID outputs and rerun just those jobs."
	)
	parser.add_argument(
		"--models-root",
		type=Path,
		default=DEFAULT_MODELS_ROOT,
		help="Root directory containing model subfolders (ignored when --model-dirs is provided).",
	)
	parser.add_argument(
		"--model-dirs",
		type=Path,
		nargs="+",
		default=None,
		help="Explicit model directories to inspect; overrides --models-root scan.",
	)
	parser.add_argument(
		"--tables-dir",
		type=Path,
		default=None,
		help="Directory containing partial FID CSVs (defaults to <models_root>-tables).",
	)
	parser.add_argument(
		"--task-ids",
		type=int,
		nargs="+",
		default=None,
		help="Optional subset of task ids to inspect.",
	)
	parser.add_argument(
		"--max-task-id",
		type=int,
		default=7,
		help="Highest task id to inspect when --task-ids is omitted.",
	)
	parser.add_argument(
		"--submit-script",
		type=Path,
		default=DEFAULT_SUBMIT_SCRIPT,
		help="Path to submit_eval_final_fid_jobs.py (relative to repository root by default).",
	)
	parser.add_argument(
		"--python-exec",
		type=Path,
		default=Path(sys.executable),
		help="Python executable used to invoke the submit script.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only print planned commands without submitting jobs.",
	)
	parser.add_argument(
		"--submit-args",
		nargs=argparse.REMAINDER,
		help="Extra arguments forwarded to the submit script (prefix with -- to terminate this parser).",
	)
	return parser.parse_args()


def resolve_model_dirs(models_root: Path, explicit: Sequence[Path] | None) -> List[Path]:
	if explicit:
		dirs: List[Path] = []
		for raw_dir in explicit:
			dir_path = raw_dir.expanduser()
			if not dir_path.is_dir():
				print(f"Warning: {dir_path} not found; skipping")
				continue
			dirs.append(dir_path)
		return dirs

	models_root = models_root.expanduser()
	if not models_root.exists():
		raise SystemExit(f"models_root not found: {models_root}")
	return sorted([p for p in models_root.iterdir() if p.is_dir()])


def determine_tasks(max_task_id: int, task_ids: Sequence[int] | None) -> List[int]:
	if task_ids:
		return sorted({int(t) for t in task_ids})
	return list(range(max_task_id + 1))


def find_missing_partials(
	run_dirs: Sequence[Path],
	tasks: Sequence[int],
	partials_dir: Path,
) -> Dict[Path, List[int]]:
	missing: Dict[Path, List[int]] = {}
	partials_available = partials_dir.exists()
	for run_dir in run_dirs:
		run_name = run_dir.name
		missing_tasks: List[int] = []
		for task_id in tasks:
			if not partials_available:
				missing_tasks.append(task_id)
				continue
			fid_partial = partials_dir / f"{run_name}_task{task_id}_fid.csv"
			avg_partial = partials_dir / f"{run_name}_task{task_id}_avg.csv"
			if not fid_partial.exists() or not avg_partial.exists():
				missing_tasks.append(task_id)
		if missing_tasks:
			missing[run_dir] = missing_tasks
	return missing


def main() -> None:
	args = parse_args()

	models_root = args.models_root.expanduser()
	tables_dir = args.tables_dir.expanduser() if args.tables_dir else Path(f"{models_root}-tables")
	partials_dir = tables_dir / "partials"

	run_dirs = resolve_model_dirs(models_root, args.model_dirs)
	if not run_dirs:
		print("No model directories discovered. Nothing to do.")
		return

	tasks = determine_tasks(args.max_task_id, args.task_ids)
	missing = find_missing_partials(run_dirs, tasks, partials_dir)
	if not missing:
		print("All requested FID outputs are present. Nothing to rerun.")
		return

	python_exec = str(args.python_exec)
	repo_root = Path(__file__).resolve().parent
	submit_script = args.submit_script.expanduser()
	if not submit_script.is_absolute():
		submit_script = (repo_root / submit_script).resolve()
	if not submit_script.exists():
		raise SystemExit(f"Submit script not found at {submit_script}")

	for run_dir, task_list in missing.items():
		print(f"Model: {run_dir}")
		print(f"  Missing tasks: {task_list}")
		cmd = [
			python_exec,
			str(submit_script),
			"--model-dirs",
			str(run_dir),
			"--task-ids",
			*[str(t) for t in task_list],
			"--tables_dir",
			str(tables_dir),
		]
		if args.submit_args:
			cmd.extend(args.submit_args)

		cmd_repr = " ".join(cmd)
		if args.dry_run:
			print(f"[DRY RUN] Would run: {cmd_repr}")
			continue

		print(f"Submitting reruns via: {cmd_repr}")
		subprocess.run(cmd, check=True)


if __name__ == "__main__":
	main()
