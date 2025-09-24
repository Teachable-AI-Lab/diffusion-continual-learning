#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import Optional, Tuple

import torch

# Minimal replica of experiment naming used by training script

def make_exp_path_from_config(cfg: dict, args) -> Path:
    out_dir = Path(cfg.get("output_dir", "./experiments"))
    dataset = cfg["dataset"]
    
    use_ewc = bool(cfg.get("use_ewc", False))
    fisher = "-" + cfg.get("ewc_fisher_type", "") if use_ewc else ""
    use_gr = bool(cfg.get("use_generative_replay", False))
    use_distil = bool(cfg.get("use_distillation", False))
    seed = cfg.get("seed", 0)
    # Match training script naming: dataset[-fisher][-gr][-distill]-seed
    if args.wandb == "true":
        name = f"{dataset}-{cfg.get('ewc_fisher_type', '')}{'-gr' if use_gr else ''}{'-distil' if use_distil else ''}-{seed}"
    else:
        name = f"{dataset}{fisher}{'-gr' if use_gr else ''}{'-distill' if use_distil else ''}-{seed}"
    exp_path = out_dir / name
    return exp_path


def final_checkpoint_exists(exp_path: Path, cfg: dict) -> bool:
    # consider either explicit final snapshot or the last per-task snapshot
    final_model = exp_path / "final_model.pt"
    if final_model.exists():
        return True
    return False
    # num_classes = int(cfg["num_classes"])
    # group_size = int(cfg["group_size"])
    # n_tasks = (num_classes + group_size - 1) // group_size
    # last_task_ckpt = exp_path / f"model-task{n_tasks-1}.pt"
    # return last_task_ckpt.exists()


def _scan_max_task_from_files(exp_path: Path) -> int:
    if not exp_path.exists():
        return -1
    max_task = -1
    for p in exp_path.glob("model-task*.pt"):
        m = re.match(r"model-task(\d+)\.pt$", p.name)
        if m:
            try:
                t = int(m.group(1))
                if t > max_task:
                    max_task = t
            except Exception:
                pass
    return max_task


def read_current_progress(exp_path: Path) -> Tuple[int, int]:
    """Return (task_id, epoch) from latest.pt if available; otherwise infer from per-task files.
    If nothing exists, returns (-1, -1).
    """
    latest = exp_path / "latest.pt"
    cur_task, cur_epoch = -1, -1
    if latest.exists():
        try:
            ckpt = torch.load(latest, map_location="cpu")
            cur_task = int(ckpt.get("task_id", -1))
            cur_epoch = int(ckpt.get("epoch", -1))
        except Exception:
            # ignore unreadable latest; fall back to scan
            pass
    if cur_task < 0:
        # fallback to max task file present
        mt = _scan_max_task_from_files(exp_path)
        if mt >= 0:
            cur_task, cur_epoch = mt, 0
    return cur_task, cur_epoch


def progress_cmp(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compare two (task, epoch) tuples. Returns -1 if a<b, 0 if equal, 1 if a>b."""
    if a[0] != b[0]:
        return -1 if a[0] < b[0] else 1
    if a[1] != b[1]:
        return -1 if a[1] < b[1] else 1
    return 0


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            with state_path.open("r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "resubmit_count": 0,
        "last_progress": None,  # dict with keys task, epoch
        "last_progress_repeat_count": 0,
    }


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f)
    tmp.replace(state_path)


def main():
    p = argparse.ArgumentParser(description="Check for final model and (re)submit if missing")
    p.add_argument("--config", required=True, help="Path to JSON config used for training")
    p.add_argument("--wandb", choices=["true", "false"], default="false", help="Use wandb run variant (selects train-model-embers.py)")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[CHECK] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    # Load config JSON
    try:
        with cfg_path.open("r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[CHECK] Failed to load config JSON: {e}", file=sys.stderr)
        sys.exit(2)

    exp_path = make_exp_path_from_config(cfg, args)
    print(f"[CHECK] Experiment path: {exp_path}")

    if final_checkpoint_exists(exp_path, cfg):
        print("[CHECK] Final checkpoint found. Nothing to do.")
        sys.exit(0)

    # Not finished: decide whether to resubmit based on progress/state
    state_path = exp_path / "resubmit_state.json"
    state = load_state(state_path)
    cur_progress = read_current_progress(exp_path)
    last_progress_tup: Optional[Tuple[int, int]] = None
    if isinstance(state.get("last_progress"), dict):
        lp = state["last_progress"]
        last_progress_tup = (int(lp.get("task", -1)), int(lp.get("epoch", -1)))

    print(f"[CHECK] Current progress: task={cur_progress[0]}, epoch={cur_progress[1]}")
    if last_progress_tup is not None:
        print(f"[CHECK] Last recorded progress: task={last_progress_tup[0]}, epoch={last_progress_tup[1]}, repeats={state.get('last_progress_repeat_count', 0)}")

    # Evaluate progress movement
    stagnant_repeats = int(state.get("last_progress_repeat_count", 0))
    stop_due_to_regress = False
    stop_due_to_stagnant_twice = False
    if last_progress_tup is not None:
        cmp = progress_cmp(cur_progress, last_progress_tup)
        if cmp < 0:
            stop_due_to_regress = True
        elif cmp == 0:
            stagnant_repeats += 1
            if stagnant_repeats >= 2:
                stop_due_to_stagnant_twice = True
        else:  # progress improved
            stagnant_repeats = 0

    # Update state with current progress regardless of decision
    state["last_progress"] = {"task": int(cur_progress[0]), "epoch": int(cur_progress[1])}
    state["last_progress_repeat_count"] = stagnant_repeats

    if stop_due_to_regress:
        print("[CHECK] Progress regressed compared to last run; not resubmitting.")
        save_state(state_path, state)
        sys.exit(0)

    if stop_due_to_stagnant_twice:
        print("[CHECK] Progress unchanged for two consecutive failures; not resubmitting.")
        save_state(state_path, state)
        sys.exit(0)

    # Enforce max resubmissions
    submissions = int(state.get("resubmit_count", 0))
    if submissions >= 20:
        print(f"[CHECK] Reached max resubmissions ({submissions}/20). Not resubmitting.")
        save_state(state_path, state)
        sys.exit(0)

    # Not finished and allowed: submit training + chained check
    script_dir = Path("/storage/home/hcoda1/1/agupta886/scratch/diffusion-continual-learning/tmp")
    submit_sh = script_dir / "submit_train_and_check.sh"
    if not submit_sh.exists():
        print(f"[CHECK] Submission helper not found: {submit_sh}", file=sys.stderr)
        sys.exit(3)

    print("[CHECK] Final checkpoint missing. Submitting training + follow-up check...")
    try:
        cmd = ["bash", str(submit_sh), str(cfg_path), args.wandb]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        # Increment submission count and persist state
        state["resubmit_count"] = int(state.get("resubmit_count", 0)) + 1
        save_state(state_path, state)
    except subprocess.CalledProcessError as e:
        print(f"[CHECK] Submission failed: {e}\n{e.stdout}\n{e.stderr}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
