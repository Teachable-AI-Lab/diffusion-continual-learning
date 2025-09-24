import argparse
import os
from pathlib import Path
import random
from typing import Optional, Tuple, List
import csv

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import src.utils as utils
from src.ddim import build_conditional_ddim
from src.parameter_scoring import (
    compute_rank1_coeff_and_mean,
    compute_top_eigenpair_two_pass,
)
from src.gr import GenerativeReplay


# --------------- Local helpers (self-contained) --------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    # numpy seeding is done in utils.set_seed in other files; keep minimal here
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def evaluate_fid(model, test_loader, device) -> float:
    fid_eval = utils.FIDEvaluator(device=device)
    return fid_eval.fid_loader_vs_model(test_loader, model)


def freeze_model(model):
    return utils.freeze_model(model)


class EWC:
    """
    Lightweight EWC implementation with same API as src.ewc.EWC, defined locally
    so we can construct from saved teacher checkpoints without changing other files.
    Stores tasks as tuples of (theta0_flat, mu, c, diag), and computes quadratic penalty.
    """
    def __init__(self, fisher_type: str):
        self.fisher_type = fisher_type  # "diag" | "rank1" | "rank1_opt" | "top_eig"
        self.tasks: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float], Optional[torch.Tensor]]] = []

    @staticmethod
    def _flat_unet_params(m) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in m.unet.parameters()])

    def add_task(self, teacher_model, *, mu=None, c=None, diag=None):
        theta0 = self._flat_unet_params(teacher_model.eval()).detach().cpu()
        # store CPU to reduce VRAM; move on use
        if isinstance(c, torch.Tensor):
            c = float(c.detach().cpu().item())
        self.tasks.append((theta0, None if mu is None else mu.detach().cpu(), c, None if diag is None else diag.detach().cpu()))

    def loss(self, model) -> torch.Tensor:
        theta = self._flat_unet_params(model)
        total = torch.zeros((), device=theta.device)
        for (theta0, mu, c, diag) in self.tasks:
            theta0_d = theta0.to(theta)
            delta = theta - theta0_d
            if self.fisher_type == "diag":
                d = diag.to(theta)
                total = total + 0.5 * (d * (delta * delta)).sum()
            else:
                m = mu.to(theta)
                proj = (m * delta).sum()
                if self.fisher_type in ("rank1_opt", "top_eig"):
                    total = total + 0.5 * float(c) * (proj * proj)
                else:  # "rank1"
                    total = total + 0.5 * (proj * proj)
        return total


def make_exp_path(args) -> Path:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = args.dataset
    if getattr(args, 'use_ewc', False):
        name += f"-{args.ewc_fisher_type}"
    if getattr(args, 'use_generative_replay', False):
        name += "-gr"
    if getattr(args, 'use_distillation', False):
        name += "-distill"
    name += f"-{args.seed}"
    exp_path = root / name
    exp_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment path: {exp_path}")
    return exp_path


def save_latest_ckpt(latest_path: Path, model, optimizer, task_id: int, epoch: int):
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'task_id': task_id,
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, latest_path)
    print(f"Saved latest checkpoint: task_id={task_id}, epoch={epoch}, path={latest_path}")


def load_latest_ckpt(latest_path: Path, model, optimizer, device) -> Tuple[int, int]:
    if not latest_path.exists():
        print(f"No latest checkpoint found at {latest_path}. Starting fresh at task 0, epoch 0.")
        return 0, 0
    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_task = int(ckpt.get('task_id', 0))
    start_epoch = int(ckpt.get('epoch', 0)) + 1
    print(f"Loaded latest checkpoint from {latest_path}: saved_task={ckpt.get('task_id', 'NA')}, saved_epoch={ckpt.get('epoch', 'NA')} -> start_task={start_task}, start_epoch={start_epoch}")
    return start_task, start_epoch



def build_ewc_from_history(
    model,
    fisher_type: str,
    exp_path: Path,
    upto_task_exclusive: int,
    device,
) -> Optional[EWC]:
    """
    Rebuild EWC using saved per-task model checkpoints and fisher tensors.
    Expects files:
      - model-task{t}.pt
      - fisher-task{t}.pt  -> tuple (c, mu, diag) depending on fisher_type
    for t in 0..upto_task_exclusive-1
    """
    if upto_task_exclusive <= 0:
        return None
    ewc = EWC(fisher_type)
    print(f"Rebuilding EWC up to task {upto_task_exclusive} (exclusive). Fisher type: {fisher_type}")
    for t in range(upto_task_exclusive):
        model_path = exp_path / f"model-task{t}.pt"
        fisher_path = exp_path / f"fisher-task{t}.pt"
        if not (model_path.exists() and fisher_path.exists()):
            # Issue granular warnings and skip
            if not model_path.exists():
                print(f"Warning: missing model checkpoint for task {t}: {model_path}. Skipping this task in EWC.")
            if not fisher_path.exists():
                print(f"Warning: missing Fisher file for task {t}: {fisher_path}. EWC will not include this task.")
            continue
        # load teacher weights onto a fresh copy of the same architecture
        print(f"EWC: loading teacher for task {t} from {model_path}")
        teacher = build_conditional_ddim(
            in_channel=model.in_channels,
            image_size=model.image_size,
            num_class_labels=model.num_class_labels,
            ewc_lambda=model.ewc_lambda,
            gr_kl=model.gr_kl,
        ).to(device)
        teacher.load_state_dict(torch.load(model_path, map_location=device))
        teacher = freeze_model(teacher)
        c, mu, diag = torch.load(fisher_path, map_location='cpu')
        mu_info = None if mu is None else f"len={mu.numel()}"
        diag_info = None if diag is None else f"len={diag.numel()}"
        print(f"EWC: adding task {t} with fisher from {fisher_path}: c={c}, mu={mu_info}, diag={diag_info}")
        ewc.add_task(teacher, mu=mu, c=c, diag=diag)
        # free teacher
        del teacher
        torch.cuda.empty_cache()
    print(f"EWC rebuilt with {len(ewc.tasks)} tasks included.")
    return ewc if len(ewc.tasks) > 0 else None


def build_gr_for_task(
    model,
    exp_path: Path,
    task_id: int,
    group_size: int,
    args,
    device,
) -> Optional[GenerativeReplay]:
    """
    Build GR teacher for current task training (i.e., from previous task's model).
    Returns None for task 0 or if use_generative_replay is False.
    """
    if not getattr(args, 'use_generative_replay', False):
        print("GR disabled by config. Not building GR teacher.")
        return None
    if task_id <= 0:
        print(f"GR not applicable for task {task_id}. Returning None.")
        return None
    prev_model_path = exp_path / f"model-task{task_id-1}.pt"
    if not prev_model_path.exists():
        # cannot build; maybe first run of this task
        print(f"GR: previous model missing at {prev_model_path}. Skipping GR for task {task_id}.")
        return None
    print(f"GR: loading teacher for task {task_id} from {prev_model_path}")
    teacher = build_conditional_ddim(
        in_channel=model.in_channels,
        image_size=model.image_size,
        num_class_labels=model.num_class_labels,
        ewc_lambda=model.ewc_lambda,
        gr_kl=model.gr_kl,
    ).to(device)
    teacher.load_state_dict(torch.load(prev_model_path, map_location=device))
    teacher = freeze_model(teacher)

    old_classes = list(range(task_id * group_size))
    print(f"GR: old_classes count={len(old_classes)} for task {task_id}")
    # If a saved pool exists for this task, create a tiny pool then override from disk
    pool_file = exp_path / f"gr-pool-task{task_id}.pt"
    has_saved_pool = pool_file.exists()
    print(f"GR: pool file for task {task_id}: {pool_file} (exists={has_saved_pool})")
    gr = GenerativeReplay(
        teacher,
        old_classes=old_classes,
        alpha=args.gr_alpha,
        batch_size=args.batch_size,
        pool_size_per_class=(1 if has_saved_pool else args.gr_pool_size_per_class),
        num_inference_steps=args.gr_num_inference_steps,
        eta=args.gr_eta,
        seed=args.seed,
        device=device,
    )
    if has_saved_pool:
        try:
            state = torch.load(pool_file, map_location=device)
            gr.pool = state['pool'].to(device)
            gr.pool_labels = state['pool_labels'].to(device)
            gr.pool_size = int(gr.pool.shape[0])
            gr.pool_indices = torch.randperm(gr.pool_size, device=device)
            gr.pool_ptr = 0
            # sanity: old_classes may need to match
            # We won't strictly enforce; but we can log size info
            print(f"Loaded GR pool for task {task_id} from {pool_file} with {gr.pool_size} samples.")
        except Exception as e:
            print(f"Warning: failed to load GR pool from {pool_file}: {e}. Using freshly built pool.")
    else:
        print(f"GR: created fresh pool for task {task_id} with pool_size_per_class={args.gr_pool_size_per_class}")
    return gr


def train_one_task_with_resume(
    model,
    train_loader,
    task_id: int,
    optimizer,
    ewc,  # may be None
    gr,   # may be None
    kl: bool,
    num_epochs: int,
    exp_path: Path,
    device,
    start_epoch: int,
    latest_ckpt_path: Path,
    arch_dir: Path,
):
    """
    Train a single task with EWC + GR + optional KL distill. Supports resuming at epoch granularity.
    Saves rolling latest.pt each epoch and archives periodically.
    """
    model.to(device)
    model.train()

    print(f"Train task {task_id}: start_epoch={start_epoch}, end_epoch={num_epochs-1}")

    for epoch in tqdm(range(start_epoch, num_epochs), desc=f"Task {task_id} - Epoch"):
        print(f"[Task {task_id}] Epoch {epoch} starting...")
        running_loss = 0.0
        running_ddim = 0.0
        running_ewc = 0.0
        running_kl = 0.0
        num_batches = 0

        printed_batch_debug = False

        for images, labels in tqdm(train_loader, leave=False, desc='Batch'):
            images = images.to(device)
            labels = labels.to(device)

            if gr is not None:
                x_old, y_old = gr.replay()
                images = torch.cat([images, x_old], dim=0)
                labels = torch.cat([labels, y_old], dim=0)

            optimizer.zero_grad(set_to_none=True)
            timesteps, noise, noisy_images, model_pred = model.diffusion_loss(images, labels)

            if gr is not None:
                replay_size = x_old.size(0)
                # split batch into new vs replay parts
                t_replay = timesteps[-replay_size:]
                noise_replay = noise[-replay_size:]
                noisy_images_replay = noisy_images[-replay_size:]
                model_pred_replay = model_pred[-replay_size:]

                t_batch = timesteps[:-replay_size]
                noise_batch = noise[:-replay_size]
                model_pred_batch = model_pred[:-replay_size]

                ddim_loss = F.mse_loss(model_pred_batch, noise_batch, reduction="mean")
            else:
                ddim_loss = F.mse_loss(model_pred, noise, reduction="mean")

            loss = ddim_loss

            loss_ewc = torch.zeros((), device=device)
            if ewc is not None and getattr(model, 'ewc_lambda', 0.0) > 0:
                loss_ewc = ewc.loss(model)
                loss = loss + model.ewc_lambda * loss_ewc

            loss_kl = torch.zeros((), device=device)
            if kl and gr is not None and getattr(model, 'gr_kl', 0.0) > 0:
                with torch.no_grad():
                    eps_teacher = gr.teacher.unet(noisy_images_replay, t_replay, y_old).sample
                eps_student = model_pred_replay
                loss_kl = F.mse_loss(eps_student, eps_teacher)
                loss = loss + model.gr_kl * loss_kl

            if not printed_batch_debug:
                print(f"[Task {task_id}] Epoch {epoch} first-batch debug:")
                print(f"  batch_total={images.shape[0]}, new_part={(0 if gr is None else images.shape[0]-replay_size)}, replay_part={(0 if gr is None else replay_size)}")
                print(f"  ddim_loss={float(ddim_loss.detach().item()):.6f}, ewc_in_use={(ewc is not None and getattr(model,'ewc_lambda',0.0)>0)}, kl_in_use={(kl and gr is not None and getattr(model,'gr_kl',0.0)>0)}")
                if ewc is not None:
                    print(f"  ewc_lambda={getattr(model,'ewc_lambda', 0.0)}")
                if gr is not None:
                    print(f"  gr_kl={getattr(model,'gr_kl', 0.0)}")
                printed_batch_debug = True

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ddim += float(ddim_loss.detach().item())
            running_ewc += float(loss_ewc.detach().item()) if ewc is not None else 0.0
            running_kl += float(loss_kl.detach().item()) if (kl and gr is not None) else 0.0
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        avg_ddim = running_ddim / max(num_batches, 1)
        avg_ewc = running_ewc / max(num_batches, 1)
        avg_kl = running_kl / max(num_batches, 1)

        print(f"[Task {task_id}] Epoch {epoch} finished. Batches={num_batches}, avg_total_loss={avg_loss:.6f}, avg_ddim={avg_ddim:.6f}, avg_ewc={avg_ewc:.6f}, avg_kl={avg_kl:.6f}")

        # Save rolling latest every epoch and archive periodically/finally
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            save_latest_ckpt(latest_ckpt_path, model, optimizer, task_id, epoch)
            print(f"[Task {task_id}] Saved latest checkpoint at epoch {epoch}.")

def main():
    parser = argparse.ArgumentParser(description="Continual Learning with Diffusion (resumable for EMBERS)")
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    initial_args = parser.parse_args()

    # Load config
    args = utils.load_config_from_json(initial_args.config)
    print(f"Loaded config from {initial_args.config}: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")
    set_seed(args.seed)

    exp_path = make_exp_path(args)
    latest_ckpt_path = exp_path / 'latest.pt'
    arch_dir = exp_path / 'checkpoints'
    print(f"Latest ckpt path: {latest_ckpt_path}")
    print(f"Archive dir: {arch_dir}")

    # Prepare FID CSV logging paths (like fid_evaluation.py)
    tables_dir = Path("~/scratch/diffusion-continual-learning/tables_and_figures").expanduser()
    fid_out_dir = tables_dir / "fid_triangles"
    fid_out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(getattr(args, 'seed', 0))
        # Include seed in method naming; mirror earlier behavior where no flags -> "-<seed>"
    _parts = []
    if getattr(args, 'use_ewc', False):
        _parts.append(str(args.ewc_fisher_type))
    if getattr(args, 'use_generative_replay', False):
        _parts.append('gr')
    if getattr(args, 'use_distillation', False):
        _parts.append('distill')
    _core = '-'.join(_parts)
    method_name = f"{_core}-{seed}"
    dataset_name = args.dataset
    n_tasks_planned = (args.num_classes + args.group_size - 1) // args.group_size
    fid_header = ["task_id"] + [f"fid-task{i}" for i in range(n_tasks_planned)]
    # method_name already includes seed; keep CSV name consistent but not duplicated
    seed_csv = fid_out_dir / f"{dataset_name}_{method_name}.csv"

    def ensure_seed_csv(path: Path, header: List[str]):
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()

    def append_row(path: Path, header: List[str], row: dict):
        ensure_seed_csv(path, header)
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

    def recompute_and_upsert_avg(seed_val: int, seed_csv_path: Path):
        avg_file = tables_dir / "avg_fid_summary.csv"
        all_task_ids = list(range(n_tasks_planned))

        if not seed_csv_path.exists():
            return
        per_task_avgs = {}
        with seed_csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row.get("task_id", -1))
                except Exception:
                    continue
                if tid < 0:
                    continue
                total = 0.0
                count = 0
                for i in range(tid + 1):
                    key = f"fid-task{i}"
                    if key in row and row[key] not in ("", None):
                        try:
                            total += float(row[key])
                            count += 1
                        except Exception:
                            pass
                per_task_avgs[tid] = (total / count) if count > 0 else ""

        base_cols = ["dataset", "method", "seed"]
        required_avg_cols = [f"avg_fid@task{i}" for i in all_task_ids]
        rows_existing = []
        existing_fieldnames = []
        if avg_file.exists():
            with avg_file.open("r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_fieldnames = list(reader.fieldnames)
                    rows_existing = list(reader)
        header_avg = existing_fieldnames[:] if existing_fieldnames else base_cols[:]
        for c in base_cols:
            if c not in header_avg:
                header_avg.append(c)
        for c in required_avg_cols:
            if c not in header_avg:
                header_avg.append(c)

        key_dataset = dataset_name
        key_method = method_name
        key_seed = str(seed_val)
        updated = False
        for r in rows_existing:
            if r.get("dataset") == key_dataset and r.get("method") == key_method and str(r.get("seed")) == key_seed:
                for i in range(n_tasks_planned):
                    col = f"avg_fid@task{i}"
                    r[col] = per_task_avgs.get(i, r.get(col, ""))
                updated = True
                break
        if not updated:
            new_row = {k: "" for k in header_avg}
            new_row["dataset"] = key_dataset
            new_row["method"] = key_method
            new_row["seed"] = key_seed
            for i in range(n_tasks_planned):
                new_row[f"avg_fid@task{i}"] = per_task_avgs.get(i, "")
            rows_existing.append(new_row)
        tmp_path = avg_file.with_suffix(".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_avg)
            writer.writeheader()
            for r in rows_existing:
                for c in header_avg:
                    if c not in r:
                        r[c] = ""
                writer.writerow(r)
        tmp_path.replace(avg_file)

    # Data
    cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset,
        batch_size=args.batch_size,
        normalize=args.normalize,
        greyscale=args.greyscale,
        group_size=args.group_size,
        n_classes=args.num_classes,
    )
    print(f"Dataset: {args.dataset}, group_size={args.group_size}, num_classes={args.num_classes}")
    print(f"#Tasks (train loaders) = {len(cl_train_loader)}")
    sample_img, _ = full_train_loader.dataset[0]
    channels = sample_img.shape[0]
    im_size = sample_img.shape[1]
    print(f"Sample img: channels={channels}, image_size={im_size}")

    # Model
    model = build_conditional_ddim(
        in_channel=channels,
        image_size=im_size,
        num_class_labels=args.num_classes,
        ewc_lambda=args.ewc_lambda,
        gr_kl=args.gr_kl,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: params={total_params}, ewc_lambda={getattr(model,'ewc_lambda',0.0)}, gr_kl={getattr(model,'gr_kl',0.0)}")

    # Optimizer (recreated per task unless resuming mid-task)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume if possible
    start_task, start_epoch = load_latest_ckpt(latest_ckpt_path, model, optimizer, device)
    num_tasks = len(cl_train_loader)
    # If a checkpoint captured the final epoch (epoch == epochs-1) and we applied +1 on load,
    # start_epoch may equal or exceed args.epochs. In that case, move to the next task.
    if start_epoch >= args.epochs:
        start_task = min(start_task + 1, num_tasks)
        start_epoch = 0
        print(f"Adjusting resume point: last epoch completed; moving to next task -> start_task={start_task}, start_epoch={start_epoch}")

    # If we've already finished all tasks, exit early
    if start_task >= num_tasks:
        print("All tasks completed previously; nothing to do.")
        final_path = exp_path / 'final_model.pt'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        return

    # Only iterate tasks from the resume point onward
    all_tasks = list(range(start_task, num_tasks))

    print(f"Starting/resuming training at task {start_task}, epoch {start_epoch}.")

    # Build EWC from history up to start_task (previously completed tasks)
    ewc = None
    if getattr(args, 'use_ewc', False):
        ewc = build_ewc_from_history(model, args.ewc_fisher_type, exp_path, start_task, device)

    # Build GR teacher for current task if needed
    gr = build_gr_for_task(model, exp_path, start_task, args.group_size, args, device)
    if gr is None:
        print("GR teacher: None")
    else:
        print("GR teacher: ready")
    kl = getattr(args, 'use_distillation', False)
    print(f"Distillation KL enabled: {kl}")

    for task_id in all_tasks:
        train_loader = cl_train_loader[task_id]

        # Fresh optimizer each task unless resuming mid-task
        if not (task_id == start_task and start_epoch > 0):
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print(f"New optimizer for task {task_id} with lr={args.lr}")

        # Compute start_epoch for this task
        se = start_epoch if task_id == start_task else 0
        print(f"Task {task_id}: starting from epoch {se} (total epochs={args.epochs})")

        # Train current task
        print(f"Training task {task_id} for {args.epochs - se} epochs (from epoch {se}).")
        train_one_task_with_resume(
            model,
            train_loader,
            task_id,
            optimizer,
            ewc,
            gr,
            kl,
            args.epochs,
            exp_path,
            device,
            se,
            latest_ckpt_path,
            arch_dir,
        )

        # Save per-task model
        model_path = exp_path / f"model-task{task_id}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved per-task model to {model_path}")

        # Evaluate FID on all seen tasks
        fids = []
        fid_row = {"task_id": task_id}
        for eval_task_id in range(task_id + 1):
            fid = evaluate_fid(model, cl_test_loader[eval_task_id], device)
            print(f"Task {task_id}, Eval task {eval_task_id}, FID: {fid:.3f}")
            fids.append(fid)
            fid_row[f"fid-task{eval_task_id}"] = float(fid)
        avg_fid = sum(fids) / len(fids)
        print(f"Average FID after task {task_id}: {avg_fid:.3f}")
        # Append FID row immediately for preemption safety
        ensure_seed_csv(seed_csv, fid_header)
        append_row(seed_csv, fid_header, fid_row)
        recompute_and_upsert_avg(seed, seed_csv)

        # Last task: no need to prep EWC/GR for next
        if task_id == all_tasks[-1]:
            print("Last task reached; finishing.")
            break

        # Compute and save Fisher info for this task (for future resume/rebuild)
        if getattr(args, 'use_ewc', False):
            diag = None; c = None; mu = None
            if args.ewc_fisher_type == "top_eig":
                c, mu = compute_top_eigenpair_two_pass(
                    model, train_loader, device=device, max_samples=10000, power_iters=getattr(args, 'power_iters', 1)
                )
                diag = None
            elif args.ewc_fisher_type in ("diag", "rank1_opt"):
                c, mu, diag = compute_rank1_coeff_and_mean(
                    model, train_loader, device=device, max_samples=10000
                )
                if args.ewc_fisher_type == "diag":
                    c, mu = None, None
                else:
                    diag = None
            mu_info = None if mu is None else f"len={mu.numel()}"
            diag_info = None if diag is None else f"len={diag.numel()}"
            print(f"Saving Fisher for task {task_id}: c={c}, mu={mu_info}, diag={diag_info}")
            torch.save((c, mu, diag), exp_path / f"fisher-task{task_id}.pt")

            # Update/initialize EWC object in-memory for next task
            frozen_model = freeze_model(model)
            if ewc is None:
                ewc = EWC(args.ewc_fisher_type)
            ewc.add_task(frozen_model, mu=mu, c=c, diag=diag)
            print(f"EWC updated with task {task_id}. Total EWC tasks now: {len(ewc.tasks)}")

        # Update GR teacher for next task
        if getattr(args, 'use_generative_replay', False):
            frozen_model = freeze_model(model)
            old_classes = list(range((task_id + 1) * args.group_size))
            if gr is None:
                gr = GenerativeReplay(
                    frozen_model,
                    old_classes=old_classes,
                    alpha=args.gr_alpha,
                    batch_size=args.batch_size,
                    pool_size_per_class=args.gr_pool_size_per_class,
                    num_inference_steps=args.gr_num_inference_steps,
                    eta=args.gr_eta,
                    seed=args.seed,
                    device=device,
                )
            else:
                gr.update_teacher(frozen_model, old_classes=old_classes)
                print(f"GR teacher updated for next task. old_classes_count={len(old_classes)}")

            # Persist GR pool for this "next" task so resume won't need regeneration
            try:
                pool_state = {
                    'pool': gr.pool.detach().cpu(),
                    'pool_labels': gr.pool_labels.detach().cpu(),
                }
                torch.save(pool_state, exp_path / f"gr-pool-task{task_id+1}.pt")
                print(f"Saved GR pool for task {task_id+1} with {gr.pool.shape[0]} samples.")
            except Exception as e:
                print(f"Warning: failed to save GR pool for task {task_id+1}: {e}")

        # Reset start markers for next loop
        start_task = task_id + 1
        start_epoch = 0
        save_latest_ckpt(latest_ckpt_path, model, optimizer, start_task, start_epoch)
        print(f"Prepared to start next task: start_task={start_task}, start_epoch={start_epoch}")

    # Final model snapshot
    final_path = exp_path / 'final_model.pt'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
