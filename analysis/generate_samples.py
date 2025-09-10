import argparse
from pathlib import Path
import math
import json
"""Generate class-conditional samples and store them as a Hugging Face Dataset.

Creates:
  <output-dir>/hf_dataset/        -> Arrow shards with columns: image (PIL), label (ClassLabel)
  <output-dir>/previews/class_xxxx/preview_class_xxxx.png  -> per-class preview (up to 8 images)
  <output-dir>/metadata.json      -> generation metadata
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

import src.utils as utils
from src.ddim import build_conditional_ddim
from analysis.common import set_seed

try:
    from datasets import Dataset, Features, ClassLabel, Image as HFImage
except ImportError:
    Dataset = None  # type: ignore
    Features = None  # type: ignore
    ClassLabel = None  # type: ignore
    HFImage = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Generate diffusion samples and save as HF dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--batch-size-sample", type=int, default=32)
    p.add_argument("--output-dir", type=str, default="/storage/coda1/p-cmaclellan3/0/shared/iclr-2026-samples")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def default_dataset_args(args):
    name = args.dataset.lower()
    if name == "mnist":
        args.num_classes = 10; args.samples_per_class = 6000; args.channels = 1; args.im_size = 32
    elif name == "fmnist":
        args.num_classes = 10; args.samples_per_class = 6000; args.channels = 1; args.im_size = 32
    elif name == "cifar10":
        args.num_classes = 10; args.samples_per_class = 5000; args.channels = 3; args.im_size = 32
    elif name == "imagenet64": ## it is imagenet32 but named imagenet64 in the repo
        args.num_classes = 1000; args.samples_per_class = 1282; args.channels = 3; args.im_size = 32
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args


def _save_metadata(out_dir: Path, args, total_samples: int, channels: int, im_size: int, hf_path: Path):
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "samples_per_class": args.samples_per_class,
        "total_samples": total_samples,
        "image_size": im_size,
        "channels": channels,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "batch_size_sample": args.batch_size_sample,
        "hf_dataset_path": str(hf_path),
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    args = default_dataset_args(args)

    if Dataset is None:
        raise ImportError("Install 'datasets' to enable HF dataset saving: pip install datasets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = build_conditional_ddim(
        in_channel=args.channels,
        image_size=args.im_size,
        num_class_labels=args.num_classes
    ).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warning] Missing keys:", missing); raise ValueError("Missing keys in checkpoint load")
    if unexpected:
        print("[Warning] Unexpected keys:", unexpected); raise ValueError("Unexpected keys in checkpoint load")
    model.eval()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = args.num_classes
    spp = args.samples_per_class
    total = num_classes * spp
    print(f"Generating {spp} samples per class (total {total}).")

    records: List[Dict] = []
    preview_k = 8

    for cls in range(num_classes):
        remain = spp
        class_dir = out_dir / "previews"; class_dir.mkdir(parents=True, exist_ok=True)
        batch_idx = 0
        while remain > 0:
            cur_bs = min(args.batch_size_sample, remain)
            with torch.no_grad():
                batch = model.sample(
                    batch_size=cur_bs,
                    labels=[cls] * cur_bs,
                    num_inference_steps=args.num_inference_steps,
                    device=device,
                    guidance_scale=args.guidance_scale,
                )  # (B,C,H,W) in [-1,1]
            for i in range(cur_bs):
                t_im = batch[i].detach().cpu().clamp(-1,1)
                pil = TF.to_pil_image((t_im + 1.0) * 0.5)
                records.append({"image": pil, "label": cls})

            if batch_idx == 0:  # preview
                take = min(preview_k, cur_bs)
                preview_stack = ((batch[:take] + 1.0) * 0.5).clamp(0,1)
                grid = make_grid(preview_stack, nrow=take, padding=2)
                grid_pil = TF.to_pil_image(grid.clamp(0,1))
                grid_pil.save(class_dir / f"preview_class_{cls:04d}.png")

            remain -= cur_bs
            batch_idx += 1
        print(f"Class {cls} done.")

    assert len(records) == total, "Mismatch in generated record count"

    features = Features({
        "image": HFImage(),
        "label": ClassLabel(num_classes=num_classes, names=[str(i) for i in range(num_classes)])
    })
    ds = Dataset.from_list(records, features=features)
    hf_path = out_dir / "hf_dataset"
    ds.save_to_disk(str(hf_path))
    print(f"Saved HF dataset to {hf_path}")

    _save_metadata(out_dir, args, total, args.channels, args.im_size, hf_path)
    print("Metadata written.")
    print("Done (HF dataset mode).")


if __name__ == "__main__":  # pragma: no cover
    main()
