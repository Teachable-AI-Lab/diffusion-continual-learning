#!/usr/bin/env python3
"""
Queue HF-sample generation jobs for directories matching *gr-distill*.

- Scans a models root directory for subfolders containing 'gr-distill'.
- Infers dataset from folder name as the substring before the first '-'.
- Builds output-dir as <output-root>/<folder-name>.
- Leaves checkpoint path blank (TODO) for you to fill; scripts are created regardless.
- Creates one sbatch script per matching folder (RTX 6000 GPU by default) and
  optionally submits them with --submit.

This script DOES NOT modify analysis/generate_samples.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import re
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Queue generate_samples.py across many model folders")
    p.add_argument("--models-root", default="/storage/home/hcoda1/1/agupta886/scratch/imagenet", help="Directory that contains model subfolders")
    p.add_argument("--output-root", required=False, default="/storage/coda1/p-cmaclellan3/0/shared/iclr-2026-samples",
                   help="Base directory to store outputs; per-model subdir will be appended")
    p.add_argument("--sbatch-dir", required=False, default="/storage/home/hcoda1/1/agupta886/scratch/diffusion-continual-learning/scripts/sampling_scripts",
                   help="Directory to store sbatch scripts")
    p.add_argument("--submit", default=False, action="store_true", help="If set, submits each sbatch script after creation")


    return p.parse_args()


def discover_model_dirs(models_root: Path) -> List[Path]:
    lst = []
    for p in models_root.iterdir():
        if p.is_dir():
            dataset = dataset_from_dirname(p.name)
            if dataset in {"mnist", "fmnist", "cifar10"} and "gr-distil" in p.name:
                lst.append(p)
            if dataset == "imagenet64" and "top_eig" not in p.name:
                lst.append(p)
    return lst

def dataset_from_dirname(dirname: str) -> str:
    return dirname.split("-", 1)[0]

def seed_from_dirname(dirname: str) -> int:
    seed = dirname.split("-")[-1]
    try:
        return int(seed)
    except ValueError:
        print(f"Warning: could not parse seed from directory name {dirname!r}, defaulting to 123")
        return 123



def build_sbatch_text(job_name, dataset, checkpoint, output_dir, seed = 123, batch_size = 1024, time = "2:00:00",
                       cpus_per_task = 8, mem_per_cpu: str = "16G", gpus = "a100:1",
                         account = "gts-cmaclellan3" , qos = "embers"):
    """Build sbatch script text; logs squeue/scontrol context into .out at start and end."""

    out_path = f"/storage/home/hcoda1/1/agupta886/slurm_outputs/{job_name}.out"
    err_path = f"/storage/home/hcoda1/1/agupta886/slurm_errors/{job_name}.err"

    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={out_path}
#SBATCH --error={err_path}
#SBATCH --account={account}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --qos={qos}
#SBATCH --time={time}

module load python cuda
source ~/scratch/python-envs/iclr-env/bin/activate
cd ~/scratch/diffusion-continual-learning
export PYTHONPATH=$(pwd)
"""

    # Normalize inputs and generate commands
    def _emit_line(ds, ckpt, out):
        ds_q = shlex.quote(str(ds))
        ckpt_q = shlex.quote(str(ckpt))
        out_q = shlex.quote(str(out))
        return f"python -m analysis.generate_samples --dataset {ds_q} --checkpoint {ckpt_q} --batch-size-sample {batch_size} --output-dir {out_q} --seed {seed}"

    if isinstance(dataset, str):
        sbatch_text += "\n" + _emit_line(dataset, checkpoint, output_dir) + "\n"
        return sbatch_text

    for ds, ckpt, out in zip(dataset, checkpoint, output_dir):
        sbatch_text += "\n" + _emit_line(ds, ckpt, out)
    sbatch_text += "\n"
    return sbatch_text


def create_job(name, dataset, checkpoint_path, output_root, sbatch_dir, seed, submit=False, time="2:00:00"):
    out_dir = output_root / name
    job_name = f"gen-{name}"  # keep it short
    sb_path = sbatch_dir / f"{name}.sbatch"
    content = build_sbatch_text(
                job_name=job_name,
                dataset=dataset,
                checkpoint=checkpoint_path if checkpoint_path is not None else "",
                output_dir=out_dir,
                seed=seed,
                time=time
            )
    sb_path.write_text(content)
    os.chmod(sb_path, 0o750)
    print(f"Wrote {sb_path}")

    if submit:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print(f"[skip submit] checkpoint NOT FOUND for {name}: {checkpoint_path}")
            return
        try:
            proc = subprocess.run(["sbatch", str(sb_path)], capture_output=True, text=True, check=True)
            print(proc.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {sb_path}: {e.stderr.strip()}")

def main():
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sbatch_dir = Path(args.sbatch_dir).expanduser().resolve()

    sbatch_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs(models_root)
    if not model_dirs:
        print(f"No matching directories under {models_root} (looking for 'gr-distill' in name).")
        return
    print(f"Found {len(model_dirs)} model dirs under {models_root}.")

    for d in sorted(model_dirs):
        dataset = dataset_from_dirname(d.name)
        if dataset == "imagenet64":
            checkpoint_dir = d
        else:
            checkpoint_dir = d / d.name
        checkpoint_path = None
        if checkpoint_dir.exists():
            names = os.listdir(checkpoint_dir)
            mx = -1
            for name in names:
                m = re.fullmatch(r"model-task(\d+)\.pt", name)
                if m:
                    try:
                        num = int(m.group(1))
                        if num > mx:
                            mx = num
                    except ValueError:
                        continue
            if mx >= 0:
                checkpoint_path = checkpoint_dir / f"model-task{mx}.pt"
        if dataset in {"mnist", "fmnist", "cifar10"}:
            seed = seed_from_dirname(d.name)
            create_job(d.name, dataset, checkpoint_path, output_root, sbatch_dir, seed, submit=args.submit)
        elif dataset == "imagenet64":
            for seed in [123, 234, 345]:
                job_name = f"{d.name}-{seed}"
                create_job(job_name, dataset, checkpoint_path, output_root, sbatch_dir, seed, submit=args.submit, time="8:00:00")


if __name__ == "__main__":
    main()
