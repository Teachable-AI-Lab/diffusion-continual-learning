#!/usr/bin/env python3
"""Generate SLURM jobs that evaluate all tasks for each checkpoint sequentially."""

from __future__ import annotations

import argparse
import csv
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

CHECKPOINT_ID_RE = re.compile(r"(?:step|checkpoint)_?(\d+)", re.IGNORECASE)
TASK_DIR_RE = re.compile(r"task_(\d+)$", re.IGNORECASE)


@dataclass
class SlurmConfig:
    job_time: str
    memory_per_cpu: str
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
    gpus_per_job: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one SLURM job per intermediate checkpoint so each GPU only evaluates "
            "a single checkpoint against previous tasks."
        )
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
        help="Explicit model directories to schedule; overrides --run-name/auto scan.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Shortcut for a single run directory under --models_root.",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/icml-checkpoint-tables"),
        help="Base directory where per-checkpoint CSVs are written.",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of task ids to schedule.",
    )
    parser.add_argument(
        "--expected-num-tasks",
        type=int,
        default=10,
        help="How many downstream tasks each checkpoint should cover (default: 10).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional cap on checkpoint steps per model (default: evaluate all discovered steps).",
    )
    parser.add_argument(
        "--checkpoint-skip-ids",
        type=str,
        nargs="+",
        default=["000000", "000011", "006001", "008001", "020000", "040001"],
        help="Step ids that should never be scheduled (integers, leading zeros allowed).",
    )
    parser.add_argument(
        "--gpu-types",
        nargs="+",
        default=["a100", "L40s", "h100", "h200"],
        help="GPU types to round-robin across when emitting #SBATCH --gpus lines.",
    )
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        help="GPUs requested per job (appended to the gpu type).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDIM steps passed to the eval script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed passed to the eval script.",
    )
    parser.add_argument(
        "--max_real",
        type=int,
        default=None,
        help="Optional cap on real samples used for evaluation.",
    )
    parser.add_argument("--job-time", default="1:00:00", help="Walltime per checkpoint job.")
    parser.add_argument("--memory-per-cpu", default="16G", help="Memory per CPU request.")
    parser.add_argument("--account", default="gts-cmaclellan3", help="Slurm account.")
    parser.add_argument("--qos", default="embers", help="Slurm QoS queue.")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task.")
    parser.add_argument("--nodes", type=int, default=1, help="Nodes per job.")
    parser.add_argument("--ntasks-per-node", type=int, default=1, help="Tasks per node.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/diffusion-continual-learning"),
        help="Project root used inside the job.",
    )
    parser.add_argument(
        "--python-script",
        default="eval_final_fid_checkpoints.py",
        help="Eval script path relative to --project-root.",
    )
    parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/storage/home/hcoda1/1/agupta886/scratch/python-envs/iclr-env/bin/activate"),
        help="Virtualenv activate script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for SLURM stdout files (default: <tables_dir>/slurm_outputs).",
    )
    parser.add_argument(
        "--error-dir",
        type=Path,
        default=None,
        help="Directory for SLURM stderr files (default: <tables_dir>/slurm_errors).",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("scripts/generated_checkpoint_jobs"),
        help="Directory where generated sbatch files are saved.",
    )
    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        default=True,
        help="Submit each generated sbatch file via sbatch (default).",
    )
    parser.add_argument(
        "--no-submit",
        dest="submit",
        action="store_false",
        help="Only write sbatch files without calling sbatch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Summarize planned jobs without writing files.")
    return parser.parse_args()


def sanitize(text: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]", "_", text)
    return sanitized or "ckpt"


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)



def iter_task_dirs(run_dir: Path) -> Iterable[tuple[int, Path]]:
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = TASK_DIR_RE.match(child.name)
        if match:
            yield int(match.group(1)), child
    meta_root = run_dir / "meta_updates"
    if meta_root.is_dir():
        for child in meta_root.iterdir():
            if not child.is_dir():
                continue
            match = TASK_DIR_RE.match(child.name)
            if match:
                yield int(match.group(1)), child


def _extract_checkpoint_meta(path: Path) -> tuple[str, int | None]:
    match = CHECKPOINT_ID_RE.search(path.stem)
    if match:
        step_str = match.group(1)
        try:
            return match.group(0), int(step_str)
        except ValueError:
            return match.group(0), None
    return path.stem, None


def collect_checkpoint_steps(run_dir: Path, task_filter: set[int] | None) -> list[tuple[str, int | None, Path]]:
    step_map: dict[str, tuple[str, int | None, Path]] = {}
    for task_id, task_dir in iter_task_dirs(run_dir):
        if task_filter is not None and task_id not in task_filter:
            continue
        for path in sorted(task_dir.glob("*.pt")):
            label, step_idx = _extract_checkpoint_meta(path)
            if step_idx is not None and step_idx == 0:
                continue
            key = f"{step_idx:06d}" if step_idx is not None else label
            if key in step_map:
                continue
            step_map[key] = (label, step_idx, path)
    entries = list(step_map.values())
    entries.sort(key=lambda item: (item[1] is None, item[1] if item[1] is not None else item[0]))
    return entries


def parse_skip_ids(values: Sequence[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except ValueError:
            print(f"Warning: skipping malformed checkpoint id '{value}'")
    return result


def csv_has_required_tasks(path: Path, expected_task_ids: list[int]) -> bool:
    try:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except Exception as exc:
        print(f"[warn] Failed to read {path}: {exc}")
        return False
    if row is None:
        return False
    for task_id in expected_task_ids:
        key = f"task{task_id}"
        val = row.get(key)
        if val is None or not str(val).strip():
            return False
    return True


def build_sbatch_body(
    config: SlurmConfig,
    job_name: str,
    gpu_type: str,
    job_time: str,
    models_root: Path,
    tables_dir: Path,
    run_name: str,
    ckpt_path: Path,
    label: str,
    csv_path: Path,
    num_inference_steps: int,
    seed: int,
    max_real: int | None,
    num_eval_tasks: int | None,
) -> str:
    cmd_lines = [
        f"srun python {config.python_script} \\",
        f'''    --models_root "{models_root}" \\''',
        f'''    --tables_dir "{tables_dir}" \\''',
        f'''    --checkpoint-path "{ckpt_path}" \\''',
        f'''    --checkpoint-label "{label}" \\''',
        f'''    --per-checkpoint-output "{csv_path}" \\''',
    ]
    if max_real is not None:
        cmd_lines.append(f"    --max_real {max_real} \\")
    if num_eval_tasks is not None:
        cmd_lines.append(f"    --num_eval_tasks {num_eval_tasks} \\")
    cmd_lines.append(f"    --num_inference_steps {num_inference_steps} \\")
    cmd_lines.append(f"    --seed {seed}")
    cmd_lines[-1] = cmd_lines[-1].rstrip(" \\")
    cmd_block = "\n".join(cmd_lines)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={job_time}
#SBATCH --mem-per-cpu={config.memory_per_cpu}
#SBATCH --gpus={gpu_type}:{config.gpus_per_job}
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

echo "Starting checkpoint {label} for {run_name} at $(date)"
{cmd_block}
echo "Finished checkpoint {label} for {run_name} at $(date)"
"""


def summarize_job_script(
    job_name: str,
    gpu_type: str,
    ckpt_path: Path,
    label: str,
    csv_path: Path,
    expected_tasks: list[int],
) -> str:
    task_list = ",".join(str(t) for t in expected_tasks) if expected_tasks else "<none>"
    return (
        f"Job {job_name}: checkpoint={ckpt_path.name}, label={label}, gpu={gpu_type}, "
        f"csv={csv_path.name}, eval_tasks=[{task_list}]"
    )


def determine_job_time(gpu_type: str, default: str) -> str:
    """Return SBATCH time string based on GPU type (case-insensitive).

    Mapping:
    - h200: 0:30:00
    - h100: 0:45:00
    - a100 or L40s: 1:30:00
    Otherwise return default.
    """
    g = (gpu_type or "").lower()
    if "h200" in g:
        return "0:30:00"
    if "h100" in g:
        return "0:45:00"
    if "a100" in g or "l40" in g:
        return "1:30:00"
    return default


def write_job_script(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def submit_job(script_path: Path) -> None:
    subprocess.run(["sbatch", str(script_path)], check=True)


def list_model_runs(models_root: Path) -> list[Path]:
    return sorted([p for p in models_root.iterdir() if p.is_dir()])


def main() -> None:
    args = parse_args()

    if not args.gpu_types:
        raise SystemExit("Provide at least one value via --gpu-types.")

    models_root = args.models_root.expanduser()
    tables_dir = args.tables_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else tables_dir / "slurm_outputs"
    error_dir = args.error_dir.expanduser() if args.error_dir else tables_dir / "slurm_errors"
    generated_dir = args.generated_dir.expanduser()

    slurm_cfg = SlurmConfig(
        job_time=args.job_time,
        memory_per_cpu=args.memory_per_cpu,
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
        gpus_per_job=args.gpus_per_job,
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
    elif args.run_name:
        run_dir = (models_root / args.run_name).expanduser()
        if not run_dir.is_dir():
            raise SystemExit(f"Run folder not found: {run_dir}")
        run_dirs = [run_dir]
    else:
        if not models_root.exists():
            raise SystemExit(f"models_root not found: {models_root}")
        run_dirs = list_model_runs(models_root)

    if not run_dirs:
        raise SystemExit("No model directories available to schedule. Check inputs.")

    task_filter = set(args.task_ids) if args.task_ids else None
    skip_step_ids = parse_skip_ids(args.checkpoint_skip_ids)
    expected_task_ids = list(range(args.expected_num_tasks)) if args.expected_num_tasks > 0 else []
    planned = 0

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_tables_dir = tables_dir / run_name
        run_tables_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_steps = collect_checkpoint_steps(run_dir, task_filter)
        if not checkpoint_steps:
            print(f"[skip] {run_name} has no checkpoint steps matching the current filters; skipping.")
            continue

        if args.max_checkpoints is not None:
            checkpoint_steps = checkpoint_steps[: args.max_checkpoints]

        for label, step_idx, ckpt_path in checkpoint_steps:
            if step_idx is not None and step_idx in skip_step_ids:
                continue

            csv_name = f"{sanitize(label)}.csv"
            csv_path = run_tables_dir / csv_name
            if csv_path.exists() and csv_has_required_tasks(csv_path, expected_task_ids):
                print(
                    f"[skip] {csv_path} already has task coverage {expected_task_ids}; skipping checkpoint {label} for {run_name}."
                )
                continue
            elif csv_path.exists():
                print(
                    f"[resubmit] {csv_path} missing some tasks from {expected_task_ids}; re-running checkpoint {label} for {run_name}."
                )

            gpu_type = args.gpu_types[planned % len(args.gpu_types)]
            job_name = f"fid-{sanitize(run_name)}-{sanitize(label)}"
            script_path = slurm_cfg.generated_dir / f"{job_name}.sbatch"
            planned += 1

            if args.dry_run:
                print(
                    summarize_job_script(
                        job_name,
                        gpu_type,
                        ckpt_path,
                        label,
                        csv_path,
                        expected_task_ids,
                    )
                )
                continue

            job_time = determine_job_time(gpu_type, slurm_cfg.job_time)

            script_body = build_sbatch_body(
                slurm_cfg,
                job_name,
                gpu_type,
                job_time,
                models_root,
                tables_dir,
                run_name,
                ckpt_path,
                label,
                csv_path,
                args.num_inference_steps,
                args.seed,
                args.max_real,
                args.expected_num_tasks if args.expected_num_tasks > 0 else None,
            )
            write_job_script(script_path, script_body)
            print(
                summarize_job_script(
                    job_name,
                    gpu_type,
                    ckpt_path,
                    label,
                    csv_path,
                    expected_task_ids,
                )
            )

            if args.submit:
                submit_job(script_path)

    if args.dry_run:
        if planned:
            print(f"Dry run summary: {planned} job(s) would be generated.")
        else:
            print("Dry run summary: no jobs would be generated.")
    elif planned == 0:
        print("No checkpoint jobs planned. Adjust filters or inputs.")
    else:
        print(f"Generated {planned} job script(s).")


if __name__ == "__main__":
    main()
