import argparse
import csv
import re
from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets import load_dataset

import src.utils as utils
from src.ddim import build_conditional_ddim


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate final models on current and previous tasks and save FID tables."
	)
	parser.add_argument(
		"--models_root",
		type=Path,
		default=Path("/storage/home/hcoda1/1/agupta886/scratch/icml-2026"),
		help="Root folder containing trained model subfolders.",
	)
	parser.add_argument(
		"--model-dirs",
		type=Path,
		nargs="+",
		default=None,
		help="Explicit model directories to evaluate; overrides --run_name/--models_root scan.",
	)
	parser.add_argument(
		"--run_name",
		type=str,
		default=None,
		help="Specific model run subfolder name to evaluate.",
	)
	parser.add_argument(
		"--task_id",
		type=int,
		default=None,
		help="Only evaluate a specific trained task checkpoint (deprecated; use --task-ids).",
	)
	parser.add_argument(
		"--task-ids",
		type=int,
		nargs="+",
		default=None,
		help="Optional list of trained task ids to evaluate.",
	)
	parser.add_argument(
		"--tables_dir",
		type=Path,
		default=None,
		help="Output folder to write CSV tables (default: <models_root>-tables).",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=512,
		help="Batch size for ImageNet test loaders.",
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=8,
		help="DataLoader workers.",
	)
	parser.add_argument(
		"--num_classes",
		type=int,
		default=500,
		help="Number of ImageNet classes to evaluate (first K classes).",
	)
	parser.add_argument(
		"--group_size",
		type=int,
		default=50,
		help="Group size per task (constant).",
	)
	parser.add_argument(
		"--imagenet_repo",
		type=str,
		default="benjamin-paine/imagenet-1k-32x32",
		help="HF ImageNet repo to use for validation split.",
	)
	parser.add_argument(
		"--cache_dir",
		type=str,
		default="/storage/home/hcoda1/1/agupta886/scratch/imagenet_dataset",
		help="HF datasets cache directory.",
	)
	parser.add_argument(
		"--num_inference_steps",
		type=int,
		default=50,
		help="DDIM sampling steps for FID evaluation.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=123,
		help="Seed for FID sampling.",
	)
	parser.add_argument(
		"--max_real",
		type=int,
		default=None,
		help="Optional cap on number of real images per task for faster eval.",
	)
	return parser.parse_args()


def list_model_runs(models_root: Path) -> list[Path]:
	return sorted([p for p in models_root.iterdir() if p.is_dir()])


def list_task_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
	default_pattern = re.compile(r"model-task(\d+)\.pt$")
	omaml_pattern = re.compile(r"task(\d+)_model\.pt$")
	if "omaml" in run_dir.name.lower():
		patterns: list[re.Pattern[str]] = [omaml_pattern, default_pattern]
	else:
		patterns = [default_pattern, omaml_pattern]

	task_ckpts: list[tuple[int, Path]] = []
	for path in run_dir.iterdir():
		if not path.is_file():
			continue
		for pattern in patterns:
			match = pattern.search(path.name)
			if match:
				task_id = int(match.group(1))
				task_ckpts.append((task_id, path))
				break
	return sorted(task_ckpts, key=lambda x: x[0])


def build_imagenet_test_loaders(
	batch_size: int,
	group_size: int,
	num_classes: int,
	cache_dir: str,
	hf_repo: str,
	num_workers: int,
) -> tuple[list[DataLoader], torch.utils.data.Dataset]:
	if num_classes % group_size != 0:
		raise ValueError("num_classes must be divisible by group_size.")

	print(f"Loading ImageNet validation split from {hf_repo}...")
	hf_val = load_dataset(hf_repo, split="validation", cache_dir=cache_dir)

	def keep_first_k(example):
		return int(example["label"]) < num_classes

	hf_val = hf_val.filter(keep_first_k)

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]
	)

	class HFImageNetVal(torch.utils.data.Dataset):
		def __init__(self, hf_ds, transform_fn):
			self.ds = hf_ds
			self.transform = transform_fn

		def __len__(self):
			return len(self.ds)

		def __getitem__(self, idx):
			rec = self.ds[int(idx)]
			img = rec["image"]
			y = int(rec["label"])
			x = self.transform(img)
			return x, y

	dataset = HFImageNetVal(hf_val, transform)

	indices_by_task: list[list[int]] = [[] for _ in range(num_classes // group_size)]
	for i in range(len(hf_val)):
		y = int(hf_val[i]["label"])
		if y >= num_classes:
			continue
		task_id = y // group_size
		indices_by_task[task_id].append(i)

	loaders: list[DataLoader] = []
	for task_id, indices in enumerate(indices_by_task):
		subset = Subset(dataset, indices)
		loaders.append(
			DataLoader(
				subset,
				batch_size=batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=True,
			)
		)
		print(f"Task {task_id}: {len(indices)} samples")

	return loaders, dataset


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def evaluate_fid(
	model,
	test_loader,
	device,
	num_inference_steps: int,
	seed: int,
	max_real: int | None,
) -> float:
	fid_eval = utils.FIDEvaluator(device=device)
	return fid_eval.fid_loader_vs_model(
		test_loader,
		model,
		num_inference_steps=num_inference_steps,
		seed=seed,
		max_real=max_real,
	)


def main() -> None:
	cli_args = parse_args()

	models_root = cli_args.models_root.expanduser()
	tables_dir = (
		cli_args.tables_dir.expanduser()
		if cli_args.tables_dir
		else Path(f"{models_root}-tables")
	)
	tables_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	set_seed(int(cli_args.seed))

	print("Loading datasets...")
	print(
		f"Using first {cli_args.num_classes} ImageNet classes with group size {cli_args.group_size}."
	)
	cl_test_loader, full_test_dataset = build_imagenet_test_loaders(
		batch_size=cli_args.batch_size,
		group_size=cli_args.group_size,
		num_classes=cli_args.num_classes,
		cache_dir=cli_args.cache_dir,
		hf_repo=cli_args.imagenet_repo,
		num_workers=cli_args.num_workers,
	)
	sample_x, _ = full_test_dataset[0]
	im_size = sample_x.shape[1]
	channels = sample_x.shape[0]
	print("Image shape:", sample_x.shape)

	run_dirs: list[Path] = []
	if cli_args.model_dirs:
		for raw_dir in cli_args.model_dirs:
			dir_path = raw_dir.expanduser()
			if not dir_path.is_dir():
				print(f"Skipping missing model directory: {dir_path}")
				continue
			run_dirs.append(dir_path)
	elif cli_args.run_name:
		run_dir = (models_root / cli_args.run_name).expanduser()
		if not run_dir.is_dir():
			raise SystemExit(f"Run folder not found: {run_dir}")
		run_dirs = [run_dir]
	else:
		if not models_root.exists():
			raise SystemExit(f"models_root not found: {models_root}")
		run_dirs = list_model_runs(models_root)
	if not run_dirs:
		raise SystemExit(
			"No model subfolders found. Check --model-dirs, --models_root, or --run_name."
		)

	task_filter: set[int] | None = None
	if cli_args.task_ids:
		task_filter = set(cli_args.task_ids)
	elif cli_args.task_id is not None:
		task_filter = {cli_args.task_id}

	summary_entries: list[tuple[str, dict[int, float]]] = []

	for run_dir in run_dirs:
		run_name = run_dir.name
		print(f"Evaluating run: {run_name}")

		task_ckpts = list_task_checkpoints(run_dir)
		if not task_ckpts:
			print(
				f"No checkpoint files (model-task*.pt or task*_model.pt) found in {run_dir}, skipping."
			)
			continue
		if task_filter is not None:
			task_ckpts = [t for t in task_ckpts if t[0] in task_filter]
			if not task_ckpts:
				print(
					f"No checkpoints for requested task ids {sorted(task_filter)} in {run_dir}, skipping."
				)
				continue

		model = build_conditional_ddim(
			in_channel=channels,
			image_size=im_size,
			num_class_labels=cli_args.num_classes,
			ewc_lambda=0.0,
			gr_kl=0.0,
		).to(device)

		per_task_fids: dict[int, dict[int, float]] = {}
		avg_by_task: dict[int, float] = {}

		for task_id, ckpt_path in task_ckpts:
			print(f"  Loading checkpoint: {ckpt_path.name}")
			state = torch.load(ckpt_path, map_location=device)
			model.load_state_dict(state)

			max_eval_task = min(task_id, len(cl_test_loader) - 1)
			fids: list[float] = []
			task_fids: dict[int, float] = {}
			for eval_task_id in range(max_eval_task + 1):
				fid = evaluate_fid(
					model,
					cl_test_loader[eval_task_id],
					device,
					num_inference_steps=cli_args.num_inference_steps,
					seed=cli_args.seed,
					max_real=cli_args.max_real,
				)
				fids.append(fid)
				task_fids[eval_task_id] = fid
				print(f"    Task {task_id} vs Task {eval_task_id}: FID {fid:.3f}")

			avg_fid = float(sum(fids) / max(len(fids), 1))
			avg_by_task[task_id] = avg_fid
			per_task_fids[task_id] = task_fids
			print(f"    Avg FID after task {task_id}: {avg_fid:.3f}")

		fid_csv = tables_dir / f"{run_name}_fid_by_task.csv"
		if per_task_fids:
			max_eval = max((max(results.keys()) for results in per_task_fids.values()), default=-1)
			if max_eval >= 0:
				fieldnames = ["trained_task"] + [str(i) for i in range(max_eval + 1)]
				matrix_rows: list[dict] = []
				for trained_task in sorted(per_task_fids.keys()):
					row: dict = {"trained_task": trained_task}
					results = per_task_fids[trained_task]
					for eval_task in range(max_eval + 1):
						val = results.get(eval_task)
						row[str(eval_task)] = f"{val:.6f}" if val is not None else ""
					matrix_rows.append(row)
				write_csv(fid_csv, fieldnames, matrix_rows)
				print(f"Wrote task x task matrix: {fid_csv}")
			else:
				write_csv(fid_csv, ["trained_task"], [])
				print(f"No evaluation tasks found for {run_name}; wrote empty matrix")
		else:
			write_csv(fid_csv, ["trained_task"], [])
			print(f"No checkpoints evaluated for {run_name}; wrote empty matrix")

		summary_entries.append((run_name, avg_by_task))

	if summary_entries:
		all_task_ids = sorted({task for _, avg_map in summary_entries for task in avg_map})
		fieldnames = ["model"] + [f"task{task_id}" for task_id in all_task_ids]
		summary_rows: list[dict] = []
		for model_name, avg_map in summary_entries:
			row: dict = {"model": model_name}
			for task_id in all_task_ids:
				val = avg_map.get(task_id)
				row[f"task{task_id}"] = f"{val:.6f}" if val is not None else ""
			summary_rows.append(row)
		summary_path = tables_dir / "avg_fid_summary.csv"
		write_csv(summary_path, fieldnames, summary_rows)
		print(f"Wrote global summary: {summary_path}")
	else:
		print("No evaluations completed; skipping avg summary.")


if __name__ == "__main__":
	main()
