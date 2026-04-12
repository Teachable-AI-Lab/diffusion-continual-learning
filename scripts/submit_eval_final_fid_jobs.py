#!/usr/bin/env python3
"""Generate and submit SLURM jobs that evaluate each model sequentially."""

from __future__ import annotations

import argparse
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SlurmConfig:
    job_time: str
    memory_per_cpu: str
    gpus: str
    account: str
    qos: str
    cpus_per_task: int
    nodes: int
    ntasks_per_node: int
    output_dir: Path
    error_dir: Path
    generated_dir: Path
    project_root: Path
    python_script: str
    venv_activate: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one SLURM job per model directory to run eval_final_fid.py"
    )
    parser.add_argument(
        "--models_root",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/icml-2026"),
        help="Root folder containing model run subfolders.",
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit model directories to schedule; overrides automatic scan of --models_root.",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=None,
        help="Output tables directory. Defaults to <models_root>-tables.",
    )
    parser.add_argument(
        "--max_task_id",
        type=int,
        default=None,
        help="Optional maximum task id to evaluate (inclusive). Defaults to all tasks.",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit task ids to schedule. Overrides --max_task_id filter.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDIM steps for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for sampling.",
    )
    parser.add_argument(
        "--max_real",
        type=int,
        default=None,
        help="Optional cap on real images per task.",
    )
    parser.add_argument("--job-time", default="1:30:00", help="Walltime.")
    parser.add_argument("--memory-per-cpu", default="16G", help="Memory per CPU.")
    parser.add_argument("--gpus", default="l40s:1", help="GPU request string.")
    parser.add_argument("--account", default="gts-cmaclellan3", help="Slurm account.")
    parser.add_argument("--qos", default="embers", help="Slurm QoS.")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task.")
    parser.add_argument("--nodes", type=int, default=1, help="Nodes per job.")
    parser.add_argument("--ntasks-per-node", type=int, default=1, help="Tasks per node.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/diffusion-continual-learning"),
        help="Project root to cd into before running eval.",
    )
    parser.add_argument(
        "--python-script",
        default="eval_final_fid.py",
        help="Path to eval script (relative to project root).",
    )
    parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/r-nisha3-0/iclr-env/bin/activate"),
        help="Path to virtualenv activate script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/icml-auto-slurm"),
        help="Directory for SLURM stdout files.",
    )
    parser.add_argument(
        "--error-dir",
        type=Path,
        default=None,
        help="Directory for SLURM stderr files.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("scripts/generated_eval_fid_jobs"),
        help="Directory where generated SBATCH scripts are stored.",
    )
    parser.add_argument("--submit/--no-submit", dest="submit", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", text)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def checkpoint_patterns(run_name: str) -> list[re.Pattern[str]]:
    default_pattern = re.compile(r"model-task(\d+)\.pt$")
    omaml_pattern = re.compile(r"task(\d+)_model\.pt$")
    if "omaml" in run_name.lower():
        return [omaml_pattern, default_pattern]
    return [default_pattern, omaml_pattern]


def discover_task_ids(run_dir: Path) -> set[int]:
    patterns = checkpoint_patterns(run_dir.name)
    task_ids: set[int] = set()
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        for pattern in patterns:
            match = pattern.search(path.name)
            if match:
                task_ids.add(int(match.group(1)))
                break
    return task_ids


def build_sbatch_body(
    config: SlurmConfig,
    job_name: str,
    models_root: Path,
    tables_dir: Path,
    run_dir: Path,
    run_name: str,
    task_ids: list[int] | None,
    num_inference_steps: int,
    seed: int,
    max_real: int | None,
) -> str:
    cmd_lines = [
        f"python {config.python_script} \\",
        f'''    --models_root "{models_root}" \\''',
        f'''    --tables_dir "{tables_dir}" \\''',
        f'''    --model-dirs "{run_dir}" \\''',
    ]
    if task_ids:
        tasks_str = " ".join(str(t) for t in task_ids)
        cmd_lines.append(f"    --task-ids {tasks_str} \\")
    if max_real is not None:
        cmd_lines.append(f"    --max_real {max_real} \\")
    cmd_lines.append(f"    --num_inference_steps {num_inference_steps} \\")
    cmd_lines.append(f"    --seed {seed}")
    cmd_block = "\n".join(cmd_lines)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={config.job_time}
#SBATCH --mem-per-cpu={config.memory_per_cpu}
#SBATCH --gpus={config.gpus}
#SBATCH --account="{config.account}"
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks-per-node={config.ntasks_per_node}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --qos="{config.qos}"
#SBATCH --output={config.output_dir}/{job_name}.out
#SBATCH --error={config.error_dir}/{job_name}.err

export PYTHONUNBUFFERED=TRUE
module load python cuda
source {config.venv_activate}
cd {config.project_root}
export PYTHONPATH=$(pwd)

start_ts="$(date)"
echo "Starting FID eval for {run_name} at ${{start_ts}}"
{cmd_block}
echo "Finished FID eval for {run_name} at $(date)"
"""


def write_job_script(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def submit_job(script_path: Path) -> None:
    subprocess.run(["sbatch", str(script_path)], check=True)


def main() -> None:
    args = parse_args()

    models_root = args.models_root.expanduser()
    tables_dir = (
        args.tables_dir.expanduser()
        if args.tables_dir
        else Path(f"{models_root}-tables")
    )
    output_dir = args.output_dir.expanduser() if args.output_dir else tables_dir / "slurm_outputs"
    error_dir = args.error_dir.expanduser() if args.error_dir else tables_dir / "slurm_errors"
    generated_dir = args.generated_dir.expanduser()

    slurm_cfg = SlurmConfig(
        job_time=args.job_time,
        memory_per_cpu=args.memory_per_cpu,
        gpus=args.gpus,
        account=args.account,
        qos=args.qos,
        cpus_per_task=args.cpus_per_task,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        output_dir=output_dir,
        error_dir=error_dir,
        generated_dir=generated_dir,
        project_root=args.project_root,
        python_script=args.python_script,
        venv_activate=args.venv_activate,
    )

    ensure_dirs([tables_dir, output_dir, error_dir, slurm_cfg.generated_dir])

    run_dirs: list[Path] = []
    if args.model_dirs:
        for raw_dir in args.model_dirs:
            dir_path = raw_dir.expanduser()
            if not dir_path.is_dir():
                print(f"Warning: {dir_path} not found; skipping")
                continue
            run_dirs.append(dir_path)
    else:
        if not models_root.exists():
            raise SystemExit(f"models_root not found: {models_root}")
        run_dirs = sorted([p for p in models_root.iterdir() if (p.is_dir() and 'vrmcl' in p.name.lower())])
    if not run_dirs:
        raise SystemExit("No model directories available to schedule. Check inputs.")

    task_filter: set[int] | None = set(args.task_ids) if args.task_ids else None
    max_task_id = args.max_task_id

    planned = 0
    for run_dir in run_dirs:
        run_name = run_dir.name
        ckpt_task_ids = discover_task_ids(run_dir)

        if not ckpt_task_ids:
            print(f"[skip] {run_name} has no checkpoints; nothing to schedule.")
            continue

        all_tasks = sorted(ckpt_task_ids)
        selected_task_ids: list[int] | None = None
        if task_filter is not None:
            filtered = [t for t in all_tasks if t in task_filter]
            if not filtered:
                print(
                    f"[skip] {run_name} has no checkpoints matching requested task ids {sorted(task_filter)}."
                )
                continue
            selected_task_ids = filtered
        elif max_task_id is not None:
            filtered = [t for t in all_tasks if t <= max_task_id]
            if not filtered:
                print(f"[skip] {run_name} has no checkpoints <= task {max_task_id}.")
                continue
            selected_task_ids = filtered

        job_slug = sanitize(run_name)
        job_name = f"fid-{job_slug}"
        script_path = slurm_cfg.generated_dir / f"{job_name}.sbatch"
        planned += 1

        if args.dry_run:
            info = (
                f"tasks {selected_task_ids}" if selected_task_ids is not None else "all checkpoints"
            )
            print(f"[DRY RUN] Would create {script_path} for {run_name} ({info}).")
            continue

        script_body = build_sbatch_body(
            slurm_cfg,
            job_name,
            models_root,
            tables_dir,
            run_dir,
            run_name,
            selected_task_ids,
            args.num_inference_steps,
            args.seed,
            args.max_real,
        )
        write_job_script(script_path, script_body)
        print(f"Wrote {script_path}")

        if args.submit:
            submit_job(script_path)
            print(f"Submitted {script_path}")

    if planned == 0:
        print("No jobs planned. Check models_root and task filters.")


if __name__ == "__main__":
    main()
