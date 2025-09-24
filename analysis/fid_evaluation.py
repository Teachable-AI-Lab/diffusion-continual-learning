import argparse
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import csv
import time
from typing import Dict, List, Set


import src.utils as utils
from src.experiment_runner import evaluate_fid

from analysis.common import set_seed
from src.ddim import build_conditional_ddim


from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms
from datasets import load_dataset
import random

def get_imagenet64_test_group_loaders(
    batch_size=512,
    group_size=50,
    normalize=True,
    n_classes=1000,
    hf_repo="benjamin-paine/imagenet-1k-32x32",
    cache_dir="/storage/coda1/p-nisha3/0/shared/imagenet",
    num_workers=4,
    pin_memory=True
):
    """
    Returns:
        test_group_loaders: dict[group_id] -> DataLoader over that group's samples
        test_loader: full test DataLoader
        meta: dict with n_classes, n_groups, group_size, group_class_ranges
    """
    assert group_size > 0
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    # Load ONLY validation split
    test_hf = load_dataset(hf_repo, split="validation", cache_dir=cache_dir)

    # Transform
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    class HFImageNet64(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform, num_classes):
            self.ds = hf_ds
            self.transform = transform
            self.num_classes = int(num_classes)
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            rec = self.ds[int(idx)]
            img = rec["image"]
            y = int(rec["label"])
            x = self.transform(img)
            return x, y

    test_dataset = HFImageNet64(test_hf, transform, n_classes)

    # Grouping
    n_groups = (n_classes + group_size - 1) // group_size  # ceiling in case not divisible
    test_indices_per_group = {g: [] for g in range(n_groups)}

    for idx, (_, label) in enumerate(tqdm(test_dataset, desc="Indexing test set")):
        g = label // group_size
        if g >= n_groups:  # safety (shouldn't happen unless label==n_classes when not divisible)
            continue
        test_indices_per_group[g].append(idx)

    # Build per-group loaders
    test_group_loaders = {}
    group_items = sorted(test_indices_per_group.items())

    for g, indices in group_items:
        if not indices:
            continue
        subset = Subset(test_dataset, indices)
        test_group_loaders[g] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Full test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Metadata
    group_class_ranges = {
        g: (g * group_size, min((g + 1) * group_size - 1, n_classes - 1))
        for g in range(n_groups)
    }
    meta = dict(
        n_classes=n_classes,
        n_groups=n_groups,
        group_size=group_size,
        group_class_ranges=group_class_ranges
    )

    return test_group_loaders, test_loader, meta


def main():
    parser = argparse.ArgumentParser(description="Run Fisher analysis on saved models")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the directory containing saved model checkpoints.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("~/scratch/diffusion-continual-learning/tables_and_figures").expanduser()
    fid_out_dir = out_dir / "fid_triangles"
    fid_out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure checkpoint_dir is a Path
    checkpoint_dir = Path(args.checkpoint_dir)

    # Parse dataset/method from the last directory name: split by '-' once
    ckpt_dir_name = checkpoint_dir.name
    ds_name, method_name = ckpt_dir_name.split("-", 1)

    dataset = "imagenet64"
    cl_test_loader, _, _ = get_imagenet64_test_group_loaders(batch_size=1024)
    # cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
    #     dataset,
    #     batch_size=128,
    #     normalize=True,
    #     greyscale=False,
    #     group_size=50,
    #     n_classes=1000,
    # )
    print("Loaded CL dataset:", dataset)
    im_size = 32
    channels = 3

    # Determine tasks and available checkpoints
    all_task_ids = list(range(20))
    n_tasks = len(all_task_ids)

    print(f"[INFO] Found {n_tasks} tasks for dataset {dataset}: {all_task_ids}")
    # Helper functions for resumability
    def ensure_seed_csv(path: Path, header: List[str]):
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()

    def read_completed_tasks(path: Path) -> Set[int]:
        if not path.exists():
            return set()
        done = set()
        with path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row.get("task_id", -1))
                except Exception:
                    continue
                if tid >= 0:
                    done.add(tid)
        return done

    def append_row(path: Path, header: List[str], row: Dict):
        # Ensure header present, then append row
        ensure_seed_csv(path, header)
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

    def recompute_and_upsert_avg(seed: int, seed_csv: Path):
        avg_file = out_dir / "avg_fid_summary.csv"

        # Read rows from seed CSV to compute averages per task
        if not seed_csv.exists():
            return
        per_task_avgs: Dict[int, float] = {}
        with seed_csv.open("r") as f:
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
                    if key in row and row[key] != "" and row[key] is not None:
                        try:
                            total += float(row[key])
                            count += 1
                        except Exception:
                            pass
                per_task_avgs[tid] = (total / count) if count > 0 else ""

        # Build header and existing rows for avg file (preserve existing columns)
        base_cols = ["dataset", "method", "seed"]
        required_avg_cols = [f"avg_fid@task{i}" for i in all_task_ids]

        rows_existing: List[Dict] = []
        existing_fieldnames: List[str] = []
        if avg_file.exists():
            with avg_file.open("r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_fieldnames = list(reader.fieldnames)
                    rows_existing = list(reader)
        # Union existing header with required columns, keep order: existing first, then any missing required
        header_avg = existing_fieldnames[:] if existing_fieldnames else base_cols[:]
        for c in base_cols:
            if c not in header_avg:
                header_avg.append(c)
        for c in required_avg_cols:
            if c not in header_avg:
                header_avg.append(c)

        key_dataset = "imagenet64"
        key_method = method_name
        key_seed = str(seed)

        # Upsert row
        updated = False
        for r in rows_existing:
            if r.get("dataset") == key_dataset and r.get("method") == key_method and str(r.get("seed")) == key_seed:
                for i in all_task_ids:
                    col = f"avg_fid@task{i}"
                    r[col] = per_task_avgs.get(i, r.get(col, ""))
                updated = True
                break

        if not updated:
            new_row = {k: "" for k in header_avg}
            new_row["dataset"] = key_dataset
            new_row["method"] = key_method
            new_row["seed"] = key_seed
            for i in all_task_ids:
                new_row[f"avg_fid@task{i}"] = per_task_avgs.get(i, "")
            rows_existing.append(new_row)

        # Write atomically via temp file
        tmp_path = avg_file.with_suffix(".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_avg)
            writer.writeheader()
            for r in rows_existing:
                # ensure all columns present
                for c in header_avg:
                    if c not in r:
                        r[c] = ""
                writer.writerow(r)
        tmp_path.replace(avg_file)
        print(f"[INFO] Upserted averages for seed {seed} -> {avg_file}")

    # Load each checkpoint once, then evaluate across all seeds with resume support
    seeds = [234, 345]
    header = ["task_id"] + [f"fid-task{i}" for i in range(n_tasks)]
    seed_csv_paths = {seed: fid_out_dir / f"imagenet64_{method_name}_{seed}.csv" for seed in seeds}
    completed_by_seed: Dict[int, Set[int]] = {seed: read_completed_tasks(path) for seed, path in seed_csv_paths.items()}

    for task_id in all_task_ids:
        ckpt_path = checkpoint_dir / f"model-task{task_id}.pt"
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint {ckpt_path}, skipping row for task {task_id}")
            continue

        s = time.time()
        model = build_conditional_ddim(in_channel=channels, image_size=im_size, num_class_labels=1000).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        e = time.time()
        print(f"[INFO] Loaded model for task {task_id} from {ckpt_path} in {e - s:.2f}s")
        model.eval()

        for seed in seeds:
            # Skip if this task already completed for this seed
            if task_id in completed_by_seed.get(seed, set()):
                print(f"[RESUME] Seed {seed} already has task {task_id}, skipping.")
                continue

            set_seed(seed)
            row = {"task_id": task_id}
            for eval_task in range(task_id + 1):
                fid_val = evaluate_fid(model, cl_test_loader[eval_task], device)
                try:
                    row[f"fid-task{eval_task}"] = float(fid_val)
                except Exception:
                    row[f"fid-task{eval_task}"] = fid_val

            # Append immediately for preemption safety
            seed_csv = seed_csv_paths[seed]
            ensure_seed_csv(seed_csv, header)
            append_row(seed_csv, header, row)
            print(f"[INFO] Appended task {task_id} for seed {seed} -> {seed_csv}")

            # Update averages incrementally
            recompute_and_upsert_avg(seed, seed_csv)



if __name__ == "__main__":
    main()
