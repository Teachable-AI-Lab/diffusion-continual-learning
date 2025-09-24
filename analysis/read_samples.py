import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from datasets import load_from_disk
except ImportError as e:
    raise ImportError("Install 'datasets' to enable HF dataset reading: pip install datasets") from e


def parse_args():
    p = argparse.ArgumentParser(description="Read diffusion samples saved as a HF dataset")
    p.add_argument("--input-dir", required=True, help="Path to output dir from generate_samples.py (containing hf_dataset/ and metadata.json)")
    p.add_argument("--num-preview", type=int, default=8, help="Number of preview images to save")
    p.add_argument("--per-class-preview", action="store_true", help="Save up to --num-preview images per class")
    p.add_argument("--dump-stats", action="store_true", help="Compute and print class distribution (may be slow for large datasets)")
    p.add_argument("--preview-out", type=str, default=None, help="Directory to write preview images. Defaults to <input-dir>/read_previews")
    return p.parse_args()


def load_metadata(input_dir: Path) -> Dict:
    meta_path = input_dir / "metadata.json"
    if meta_path.is_file():
        with meta_path.open("r") as f:
            return json.load(f)
    return {}


def dataset_paths(input_dir: Path) -> Tuple[Path, Path]:
    hf_path = input_dir / "hf_dataset"
    if not hf_path.exists():
        raise FileNotFoundError(f"hf_dataset/ not found under: {input_dir}")
    preview_out = input_dir / "read_previews"
    return hf_path, preview_out


def print_summary(ds, meta: Dict):
    print("=== Dataset Summary ===")
    print(f"Rows: {len(ds)}")
    print(f"Features: {ds.features}")
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        names = ds.features["label"].names
        print(f"Num classes: {len(names)}")
    if meta:
        print("--- Metadata.json ---")
        for k in [
            "dataset",
            "num_classes",
            "samples_per_class",
            "total_samples",
            "image_size",
            "channels",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "checkpoint",
            "batch_size_sample",
        ]:
            if k in meta:
                print(f"{k}: {meta[k]}")


def save_previews(ds, out_dir: Path, k: int, per_class: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not per_class:
        take = min(k, len(ds))
        for i in range(take):
            rec = ds[i]
            pil = rec["image"]
            label = rec["label"]
            fname = out_dir / f"sample_{i:04d}_label_{label}.png"
            pil.save(fname)
        print(f"Saved {take} preview images to {out_dir}")
        return

    if "label" in ds.features and hasattr(ds.features["label"], "num_classes"):
        num_classes = ds.features["label"].num_classes
    else:
        # Fallback: discover seen labels on-the-fly
        num_classes = None

    saved_per_class: Dict[int, int] = defaultdict(int)
    total_target = 0
    if num_classes is not None:
        total_target = min(k, len(ds)) * num_classes
    else:
        total_target = len(ds)  # upper bound; we'll stop when all have k

    for idx in range(len(ds)):
        rec = ds[idx]
        label = int(rec["label"]) if isinstance(rec["label"], (int,)) else rec["label"]
        if saved_per_class[label] < k:
            pil = rec["image"]
            class_dir = out_dir / f"class_{label:04d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            pil.save(class_dir / f"sample_{saved_per_class[label]:04d}.png")
            saved_per_class[label] += 1
        # Early exit: if all classes reached k
        if num_classes is not None and len(saved_per_class) == num_classes and all(v >= k for v in saved_per_class.values()):
            break
    covered = sum(saved_per_class.values())
    print(f"Saved {covered} per-class preview images under {out_dir}")


def dump_class_stats(ds):
    print("Computing class distribution (this may take a while)...")
    ctr = Counter()
    for i in range(len(ds)):
        rec = ds[i]
        ctr[int(rec["label"])] += 1
        if (i + 1) % 100000 == 0:
            print(f"Processed {i+1} rows...")
    print("=== Class Counts ===")
    items = sorted(ctr.items())
    for lbl, cnt in items:
        print(f"label {lbl}: {cnt}")
    print(f"Total: {sum(ctr.values())}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    hf_path, default_preview_out = dataset_paths(input_dir)
    meta = load_metadata(input_dir)

    ds = load_from_disk(str(hf_path))
    print_summary(ds, meta)

    out_dir = Path(args.preview_out) if args.preview_out else default_preview_out
    save_previews(ds, out_dir, args.num_preview, args.per_class_preview)

    if args.dump_stats:
        dump_class_stats(ds)


if __name__ == "__main__":
    main()
