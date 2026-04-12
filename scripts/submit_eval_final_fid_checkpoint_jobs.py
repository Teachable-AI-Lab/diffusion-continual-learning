#!/usr/bin/env python3
"""Minimal sbatch generator for checkpoint evaluation."""

import argparse
import csv
import json
import re
import os
from pathlib import Path
import subprocess

# Constants
DEFAULT_MODELS_ROOT = Path("/storage/home/hcoda1/1/agupta886/scratch/icml-2026")
DEFAULT_TABLES_DIR = Path("/storage/home/hcoda1/1/agupta886/scratch/icml-checkpoint-tables")
DEFAULT_PROJECT_ROOT = Path("/storage/home/hcoda1/1/agupta886/scratch/diffusion-continual-learning")
DEFAULT_VENV = Path("/storage/home/hcoda1/1/agupta886/r-nisha3-0/iclr-env/bin/activate")
EXCLUDED_RUN_SUBSTRINGS = []

CHECKPOINT_ID_RE = re.compile(r"(?:step|checkpoint)_?(\d+)", re.IGNORECASE)
TASK_DIR_RE = re.compile(r"task_(\d+)$", re.IGNORECASE)
TRAINED_TASK_ID_RE = re.compile(r"(\d+)$")

def determine_job_time(gpu_type: str, default: str = "1:00:00") -> str:
    """Return SBATCH time string based on GPU type."""
    g = (gpu_type or "").lower()
    if "h200" in g: return "0:45:00"
    if "h100" in g: return "0:45:00"
    if "a100" in g: return "1:00:00"
    if "l40" in g:  return "1:30:00"
    return default

def get_checkpoint_step(path: Path):
    match = CHECKPOINT_ID_RE.search(path.stem)
    return int(match.group(1)) if match else None

def collect_checkpoints(run_dir: Path):
    """Finds all checkpoints in a run directory and groups by step."""
    groups = {}
    for child in run_dir.rglob("*.pt"):
        if "checkpoints" not in str(child) and not TASK_DIR_RE.search(str(child.parent)):
            continue
        step = get_checkpoint_step(child)
        if step is not None:
            if step not in groups: groups[step] = {}
            match = TASK_DIR_RE.search(child.parent.name)
            if match:
                task_id = int(match.group(1))
                groups[step][task_id] = child
            else:
                groups[step][0] = child # Default
    return sorted(groups.items())

def _parse_trained_task_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        match = TRAINED_TASK_ID_RE.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def is_done(csv_path: Path, task_map: dict[int, Path]):
    """Check if CSV has complete lower triangular structure for tasks 0-9."""
    if not csv_path.exists() or csv_path.stat().st_size <= 100:
        return False
    
    try:
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except Exception as exc:
        print(f"[warn] Unable to inspect CSV {csv_path}: {exc}")
        return False
    
    # Check that we have all trained_task rows 0-9
    trained_tasks_present = set()
    for row in rows:
        task_id = _parse_trained_task_id(row.get("trained_task"))
        if task_id is not None:
            trained_tasks_present.add(task_id)
    
    # Verify tasks 0-9 all exist
    if trained_tasks_present != set(range(10)):
        return False
    
    # Verify lower triangular structure: for each trained_task i, 
    # tasks 0 through i should have non-empty values
    for row in rows:
        trained_task = _parse_trained_task_id(row.get("trained_task"))
        if trained_task is None:
            continue
        
        # Check that tasks 0 through trained_task have values
        for task_idx in range(trained_task + 1):
            task_col = f"task{task_idx}"
            value = row.get(task_col, "").strip()
            if not value:  # Empty or missing value
                return False
    
    return True

def create_sbatch(job_name, run_dir, label, csv_path, args, manifest_path, gpu_type="a100", job_time="1:00:00"):
    # Determine python script path
    script_path = args.project_root / "eval_final_fid_checkpoints.py"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={job_time}
#SBATCH --gpus={gpu_type}:{args.gpus_per_job}
#SBATCH --account=gts-cmaclellan3
#SBATCH --qos=embers
#SBATCH --output={args.tables_dir}/slurm_outputs/{job_name}.out
#SBATCH --error={args.tables_dir}/slurm_errors/{job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Unset conflicting variables that can cause "srun: fatal" errors
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE || true

source {args.venv_activate}
cd {args.project_root}

echo "Starting {label}"
python {script_path} \\
    --models_root "{args.models_root}" \\
    --tables_dir "{args.tables_dir}" \\
    --checkpoint-label "{label}" \\
    --per-checkpoint-output "{csv_path}" \\
    --num_inference_steps 50 \\
    --seed 123 \\
    --task-checkpoint-manifest "{manifest_path}"

echo "Done"
"""
    return script_content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--tables_dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--venv-activate", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str)
    
    # GPU args
    parser.add_argument("--gpu-types", nargs="+", default=["L40s", "L40s", "L40s", "h200", "a100", "h200", "h200"])
    parser.add_argument("--gpus-per-job", type=int, default=1)
    parser.add_argument("--job-time", default="1:00:00")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write jobs or scripts; only print what would be queued.")
    parser.add_argument("--no-queue", action="store_true",
                        help="Submit jobs directly with sbatch instead of creating a queue.")

    # Ignored legacy args for compatibility
    parser.add_argument("--queue-max-pending", type=int, default=50)
    parser.add_argument("--submission-mode", default="sequential")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--error-dir", type=Path)
    parser.add_argument("--queue-reset", action="store_true")

    args, _ = parser.parse_known_args()

    # Sanitize environment paths
    args.models_root = args.models_root.expanduser().resolve()
    args.tables_dir = args.tables_dir.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    
    gen_dir = args.generated_dir or (args.project_root / "scripts/generated_checkpoint_jobs")
    gen_dir = gen_dir.expanduser().resolve()
    # Only create generated directories when not doing a dry run
    if not args.dry_run:
        gen_dir.mkdir(parents=True, exist_ok=True)
        (args.tables_dir / "slurm_outputs").mkdir(parents=True, exist_ok=True)
        (args.tables_dir / "slurm_errors").mkdir(parents=True, exist_ok=True)
    
    queue_path = gen_dir / "checkpoint_job_queue.json"
    jobs = []
    preview_jobs = []  # strings describing jobs that would be created (for dry-run)
    submitted_jobs = []  # Track submitted jobs when using --no-queue
    
    # Skip queue operations if --no-queue is set
    if not args.no_queue:
        if args.queue_reset and queue_path.exists():
            try:
                queue_path.unlink()
                print(f"Removed existing queue at {queue_path}")
            except OSError as exc:
                print(f"[warn] Failed to remove queue {queue_path}: {exc}")
        elif queue_path.exists():
            try:
                jobs = json.loads(queue_path.read_text()).get("jobs", [])
            except Exception as exc:
                print(f"[warn] Failed to load existing queue {queue_path}: {exc}")

    runs = [args.models_root / args.run_name] if args.run_name else list(args.models_root.glob("*"))
    planned_count = 0 
    
    for run_dir in runs:
        if not run_dir.is_dir(): continue
        
        run_name = run_dir.name
        run_name_lower = run_name.lower()
        if any(excl in run_name_lower for excl in EXCLUDED_RUN_SUBSTRINGS):
            print(f"[skip] Excluding run {run_name} due to filter")
            continue
        run_tables_dir = args.tables_dir / run_name
        run_tables_dir.mkdir(parents=True, exist_ok=True)
        
        for step, task_map in collect_checkpoints(run_dir):
            label = f"{step:06d}"
            csv_path = run_tables_dir / f"{label}.csv"
            
            if is_done(csv_path, task_map):
                continue

            job_name = f"fid_{run_name}_{label}"
            manifest_path = gen_dir / f"manifest_{run_name}_{label}.json"
            
            # Serialize path dict for script
            # eval_final_fid_checkpoints.py expects: {"task_checkpoints": [{"task_id": 1, "path": "..."}]}
            manifest_data = {
                "label": label,
                "run_name": run_name,
                "task_checkpoints": [
                    {"task_id": k, "path": str(v)} for k, v in task_map.items()
                ]
            }
            if not args.dry_run:
                manifest_path.write_text(json.dumps(manifest_data, indent=2))
            
            # Select GPU
            gpu_type = args.gpu_types[planned_count % len(args.gpu_types)]
            job_time = determine_job_time(gpu_type, args.job_time)
            planned_count += 1
            
            script_path = gen_dir / f"{job_name}.sh"
            sbatch = create_sbatch(job_name, run_dir, label, csv_path, args, manifest_path, gpu_type, job_time)
            if args.dry_run:
                # Don't write files; just record what would be done
                preview_jobs.append(f"{job_name} -> {script_path}")
                # still check duplicates against any existing jobs loaded from queue_path
                if not args.no_queue and not any(j.get("script") == str(script_path) for j in jobs):
                    jobs.append({"script": str(script_path), "status": "pending"})
            else:
                script_path.write_text(sbatch)
                script_path.chmod(0o755)
                
                if args.no_queue:
                    # Submit directly with sbatch
                    try:
                        result = subprocess.run(
                            ["sbatch", str(script_path)],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        job_id = result.stdout.strip()
                        submitted_jobs.append(f"{job_name}: {job_id}")
                        print(f"Submitted {job_name}: {job_id}")
                    except subprocess.CalledProcessError as exc:
                        print(f"[error] Failed to submit {job_name}: {exc.stderr}")
                else:
                    # Check if job already in queue (avoid duplicates)
                    if not any(j["script"] == str(script_path) for j in jobs):
                        jobs.append({"script": str(script_path), "status": "pending"})

    if args.dry_run:
        # Report what would be queued and how many new jobs
        new_count = len(preview_jobs)
        mode_str = "submitted directly" if args.no_queue else "queued"
        print(f"Dry run mode -- no files written. Jobs that would be {mode_str}:")
        for p in preview_jobs:
            print(p)
        print(f"Total jobs that would be {mode_str}: {new_count}")
    elif args.no_queue:
        # Direct submission mode
        print(f"\nSubmitted {len(submitted_jobs)} jobs directly:")
        for job_info in submitted_jobs:
            print(job_info)
    else:
        # Queue mode
        queue_path.write_text(json.dumps({"jobs": jobs}, indent=2))
        print(f"Queue updated at {queue_path}. Total jobs: {len(jobs)}")
        # Print instruction for runner
        runner = args.project_root / "scripts" / "job_queue_runner.py"
        print(f"To start runner, execute:\npython {runner} --queue {queue_path}")

if __name__ == "__main__":
    main()
