import argparse
import csv
import json
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


CHECKPOINT_ID_RE = re.compile(r"(?:step|checkpoint)_?(\d+)", re.IGNORECASE)


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
		default=4,
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
	parser.add_argument(
		"--checkpoint-path",
		type=Path,
		default=None,
		help="Evaluate a single checkpoint .pt file instead of scanning run directories.",
	)
	parser.add_argument(
		"--checkpoint-label",
		type=str,
		default=None,
		help="Optional label override for the single checkpoint output row.",
	)
	parser.add_argument(
		"--per-checkpoint-output",
		type=Path,
		default=None,
		help="CSV path for single checkpoint mode (default: tables_dir/<checkpoint>.csv).",
	)
	parser.add_argument(
		"--task-checkpoint-manifest",
		type=Path,
		default=None,
		help="JSON manifest describing per-task checkpoints for a single intermediate step.",
	)
	parser.add_argument(
		"--previous-only",
		action="store_true",
		help="Only evaluate against tasks trained before the checkpoint's task id.",
	)
	parser.add_argument(
		"--num_eval_tasks",
		type=int,
		default=None,
		help="Override the number of tasks (0-indexed) to evaluate for each checkpoint (default: all available).",
	)
	parser.add_argument(
		"--force-recompute",
		action="store_true",
		help="Ignore existing CSV progress and recompute all requested entries from scratch.",
	)
	return parser.parse_args()


def list_model_runs(models_root: Path) -> list[Path]:
	return sorted([p for p in models_root.iterdir() if p.is_dir()])


def is_omaml_run(run_dir: Path) -> bool:
	return "omaml" in run_dir.name.lower()


def list_task_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
	default_pattern = re.compile(r"model-task(\d+)\.pt$")
	omaml_pattern = re.compile(r"task(\d+)_model\.pt$")
	patterns: list[re.Pattern[str]]
	if is_omaml_run(run_dir):
		patterns = [omaml_pattern, default_pattern]
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


def _checkpoint_dir_candidates(run_dir: Path, task_id: int) -> list[Path]:
	candidates: list[Path] = []
	if is_omaml_run(run_dir):
		candidates.append(run_dir / "meta_updates" / f"task_{task_id}")
	candidates.append(run_dir / f"task_{task_id}")
	return candidates


def _extract_checkpoint_label(path: Path) -> tuple[str, int | None]:
	match = CHECKPOINT_ID_RE.search(path.stem)
	if match:
		step_str = match.group(1)
		try:
			step_idx = int(step_str)
		except ValueError:
			step_idx = None
		label = match.group(0)
		return label, step_idx
	return path.stem, None


def list_task_checkpoint_series(run_dir: Path, task_id: int) -> list[tuple[str, Path]]:
	"""Return (label, path) pairs for intermediate checkpoints of a task."""
	entries: list[tuple[int | None, str, Path]] = []
	for base in _checkpoint_dir_candidates(run_dir, task_id):
		if not base.is_dir():
			continue
		for path in sorted(base.glob("*.pt")):
			label, step_idx = _extract_checkpoint_label(path)
			if step_idx is not None and step_idx == 0:
				continue
			entries.append((step_idx, label, path))
		if entries:
			break
	if not entries:
		return []
	entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0, item[1]))
	return [(label, path) for _, label, path in entries]


def fixed_eval_task_ids(num_eval_tasks: int | None, num_loaders: int) -> list[int]:
	if num_loaders <= 0:
		return []
	max_idx = num_loaders - 1
	if num_eval_tasks is not None:
		max_idx = min(max_idx, num_eval_tasks - 1)
	if max_idx < 0:
		return []
	return list(range(max_idx + 1))


def build_eval_task_ids(trained_task_id: int, num_loaders: int, previous_only: bool) -> list[int]:
	limit = trained_task_id - 1 if previous_only else trained_task_id
	limit = min(limit, num_loaders - 1)
	if limit < 0:
		return []
	return list(range(limit + 1))


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


def read_task_matrix_csv(path: Path) -> dict[int, dict[int, float]]:
	data: dict[int, dict[int, float]] = {}
	if not path.exists():
		return data
	try:
		with path.open("r", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				raw_id = row.get("trained_task")
				if raw_id is None:
					continue
				task_str = str(raw_id).strip()
				if not task_str:
					continue
				try:
					task_id = int(task_str)
				except ValueError:
					match = re.search(r"(\d+)$", task_str)
					if not match:
						continue
					task_id = int(match.group(1))
				results: dict[int, float] = {}
				for key, value in row.items():
					if key == "trained_task" or value is None:
						continue
					key_str = str(key).strip()
					val_str = str(value).strip()
					if not val_str:
						continue
					if key_str.startswith("task"):
						index_part = key_str[4:]
					elif key_str.isdigit():
						index_part = key_str
					else:
						continue
					try:
						idx = int(index_part)
						results[idx] = float(val_str)
					except ValueError:
						continue
				data[task_id] = results
	except Exception as exc:
		print(f"[warn] Failed to read CSV {path}: {exc}")
	return data


def write_task_matrix_csv(
	path: Path,
	task_map: dict[int, dict[int, float]],
	column_prefix: str,
	row_order: list[int] | None = None,
	max_eval_override: int | None = None,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if row_order is None:
		row_order = sorted(task_map.keys())
	else:
		seen = set(row_order)
		for task_id in sorted(task_map.keys()):
			if task_id not in seen:
				row_order.append(task_id)
	if not row_order and task_map:
		row_order = sorted(task_map.keys())
	max_eval = -1
	for results in task_map.values():
		if results:
			max_eval = max(max_eval, max(results.keys()))
	if max_eval_override is not None:
		max_eval = max(max_eval, max_eval_override)
	fieldnames = ["trained_task"]
	if max_eval >= 0:
		eval_cols = [
			f"{column_prefix}{i}" if column_prefix else str(i)
			for i in range(max_eval + 1)
		]
		fieldnames.extend(eval_cols)
	rows: list[dict[str, str | int]] = []
	for task_id in row_order:
		row: dict[str, str | int] = {"trained_task": task_id}
		results = task_map.get(task_id, {})
		if max_eval >= 0:
			for eval_idx in range(max_eval + 1):
				key = f"{column_prefix}{eval_idx}" if column_prefix else str(eval_idx)
				val = results.get(eval_idx)
				row[key] = f"{val:.6f}" if val is not None else ""
		rows.append(row)
	write_csv(path, fieldnames, rows)


def read_checkpoint_table(path: Path) -> tuple[dict[str, dict[int, float]], list[str]]:
	rows: dict[str, dict[int, float]] = {}
	order: list[str] = []
	if not path.exists():
		return rows, order
	try:
		with path.open("r", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				label = row.get("checkpoint")
				if label is None:
					continue
				label = str(label).strip()
				if not label:
					continue
				order.append(label)
				entry: dict[int, float] = {}
				for key, value in row.items():
					if key is None or not key.startswith("task"):
						continue
					val_str = str(value).strip()
					if not val_str:
						continue
					try:
						idx = int(key[4:])
						entry[idx] = float(val_str)
					except ValueError:
						continue
				rows[label] = entry
	except Exception as exc:
		print(f"[warn] Failed to read checkpoint CSV {path}: {exc}")
	return rows, order


def write_checkpoint_table(
	path: Path,
	entries: dict[str, dict[int, float]],
	row_order: list[str],
	eval_task_ids: list[int],
) -> None:
	fieldnames = ["checkpoint"] + [f"task{i}" for i in eval_task_ids]
	rows: list[dict[str, str]] = []
	for label in row_order:
		row_data = entries.get(label, {})
		row: dict[str, str] = {"checkpoint": label}
		for eval_task_id in eval_task_ids:
			val = row_data.get(eval_task_id)
			row[f"task{eval_task_id}"] = f"{val:.6f}" if val is not None else ""
		rows.append(row)
	write_csv(path, fieldnames, rows)


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


def build_checkpoint_rows(
	checkpoint_entries: list[tuple[str, Path]],
	model,
	cl_test_loader: list[DataLoader],
	eval_task_ids: list[int],
	device,
	num_inference_steps: int,
	seed: int,
	max_real: int | None,
	output_path: Path,
	force_recompute: bool,
) -> bool:
	if force_recompute:
		row_map: dict[str, dict[int, float]] = {}
		row_order: list[str] = []
	else:
		row_map, row_order = read_checkpoint_table(output_path)
	updated = False
	for label, ckpt_path in checkpoint_entries:
		existing = {} if force_recompute else dict(row_map.get(label, {}))
		missing_eval_ids = eval_task_ids if force_recompute else [
			tid for tid in eval_task_ids if tid not in existing
		]
		if not missing_eval_ids:
			print(f"	[resume] Checkpoint {label} already complete; skipping.")
			continue
		print(f"	[ckpt] Loading {ckpt_path.name}")
		state = torch.load(ckpt_path, map_location=device)
		model.load_state_dict(state)
		for eval_task_id in missing_eval_ids:
			fid = evaluate_fid(
				model,
				cl_test_loader[eval_task_id],
				device,
				num_inference_steps=num_inference_steps,
				seed=seed,
				max_real=max_real,
			)
			existing[eval_task_id] = fid
			print(f"      checkpoint {label} vs task {eval_task_id}: FID {fid:.3f}")
		row_map[label] = existing
		if label not in row_order:
			row_order.append(label)
		write_checkpoint_table(output_path, row_map, row_order, eval_task_ids)
		updated = True
	return updated


def parse_task_checkpoint_manifest(manifest_path: Path) -> dict:
	try:
		payload = json.loads(manifest_path.read_text())
	except Exception as exc:
		raise SystemExit(f"Failed to parse manifest {manifest_path}: {exc}") from exc

	task_entries_raw = payload.get("task_checkpoints")
	if not task_entries_raw:
		raise SystemExit(f"Manifest {manifest_path} missing 'task_checkpoints' entries.")

	entries: list[tuple[int, Path]] = []
	for entry in task_entries_raw:
		if not isinstance(entry, dict):
			continue
		if "task_id" not in entry or "path" not in entry:
			continue
		try:
			task_id = int(entry["task_id"])
		except (TypeError, ValueError):
			continue
		ckpt_path = Path(entry["path"]).expanduser()
		entries.append((task_id, ckpt_path))

	if not entries:
		raise SystemExit(f"Manifest {manifest_path} did not contain valid task checkpoints.")

	entries.sort(key=lambda item: item[0])
	expected_ids_raw = payload.get("expected_task_ids")
	if expected_ids_raw is not None:
		try:
			expected_task_ids = sorted(int(val) for val in expected_ids_raw)
		except (TypeError, ValueError):
			expected_task_ids = None
	else:
		expected_task_ids = None

	return {
		"label": payload.get("checkpoint_label"),
		"run_name": payload.get("run_name"),
		"task_entries": entries,
		"expected_task_ids": expected_task_ids,
	}


def evaluate_manifest_checkpoint_matrix(
	cli_args: argparse.Namespace,
	manifest_path: Path,
	tables_dir: Path,
	cl_test_loader: list[DataLoader],
	device,
	channels: int,
	im_size: int,
	task_filter: set[int] | None,
) -> None:
	manifest_info = parse_task_checkpoint_manifest(manifest_path)
	manifest_label = manifest_info.get("label")
	label = cli_args.checkpoint_label or manifest_label or manifest_path.stem
	all_task_entries: list[tuple[int, Path]] = manifest_info["task_entries"]
	if task_filter is not None:
		selected_ids = set(task_filter)
		task_entries = [entry for entry in all_task_entries if entry[0] in selected_ids]
		if not task_entries:
			print(
				f"No task checkpoints from {manifest_path} matched filters {sorted(selected_ids)}; skipping."
			)
			return
	else:
		task_entries = all_task_entries

	available_task_ids = [task_id for task_id, _ in task_entries]
	expected_task_ids = manifest_info.get("expected_task_ids")
	if expected_task_ids:
		expected_task_ids = [task_id for task_id in expected_task_ids if task_id in available_task_ids]
	else:
		expected_task_ids = available_task_ids

	output_path = (
		cli_args.per_checkpoint_output.expanduser()
		if cli_args.per_checkpoint_output
		else tables_dir / f"{label}.csv"
	)
	if cli_args.force_recompute:
		per_task_fids: dict[int, dict[int, float]] = {}
	else:
		per_task_fids = read_task_matrix_csv(output_path)
	max_eval_seen = max(
		(max(results.keys()) if results else -1) for results in per_task_fids.values()
	) if per_task_fids else -1

	model = build_conditional_ddim(
		in_channel=channels,
		image_size=im_size,
		num_class_labels=cli_args.num_classes,
		ewc_lambda=0.0,
		gr_kl=0.0,
	).to(device)

	updated_any = False
	for task_id, ckpt_path in task_entries:
		if not ckpt_path.is_file():
			raise SystemExit(f"Checkpoint not found for task {task_id}: {ckpt_path}")
		existing = {} if cli_args.force_recompute else dict(per_task_fids.get(task_id, {}))
		eval_task_ids = build_eval_task_ids(task_id, len(cl_test_loader), cli_args.previous_only)
		if task_id not in per_task_fids:
			per_task_fids[task_id] = existing
		missing_eval_ids = eval_task_ids if cli_args.force_recompute else [
			tid for tid in eval_task_ids if tid not in existing
		]
		if not missing_eval_ids:
			print(f"[resume] Task {task_id} already evaluated; skipping.")
			if existing:
				max_eval_seen = max(max_eval_seen, max(existing.keys()))
			continue
		print(f"[checkpoint] Task {task_id}: evaluating {ckpt_path}")
		state = torch.load(ckpt_path, map_location=device)
		model.load_state_dict(state)
		for eval_task_id in missing_eval_ids:
			fid = evaluate_fid(
				model,
				cl_test_loader[eval_task_id],
				device,
				num_inference_steps=cli_args.num_inference_steps,
				seed=cli_args.seed,
				max_real=cli_args.max_real,
			)
			existing[eval_task_id] = fid
			print(f"	Task {task_id} vs Task {eval_task_id}: FID {fid:.3f}")
		per_task_fids[task_id] = existing
		if existing:
			max_eval_seen = max(max_eval_seen, max(existing.keys()))
		row_order = expected_task_ids if expected_task_ids else sorted(per_task_fids.keys())
		max_expected = max(expected_task_ids) if expected_task_ids else -1
		max_override = max(max_eval_seen, max_expected)
		write_task_matrix_csv(
			output_path,
			per_task_fids,
			column_prefix="task",
			row_order=row_order,
			max_eval_override=max_override if max_override >= 0 else None,
		)
		updated_any = True

	if not per_task_fids:
		print(f"No checkpoint rows were evaluated for manifest {manifest_path}; nothing to write.")
		return
	if updated_any:
		print(f"Wrote checkpoint matrix: {output_path}")
	else:
		print(f"Checkpoint matrix already complete: {output_path}")


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

	task_filter: set[int] | None = None
	if cli_args.task_ids:
		task_filter = set(cli_args.task_ids)
	elif cli_args.task_id is not None:
		task_filter = {cli_args.task_id}

	manifest_arg = cli_args.task_checkpoint_manifest
	if manifest_arg:
		manifest_path = manifest_arg.expanduser()
		if not manifest_path.is_file():
			raise SystemExit(f"Manifest not found: {manifest_path}")
		evaluate_manifest_checkpoint_matrix(
			cli_args,
			manifest_path,
			tables_dir,
			cl_test_loader,
			device,
			channels,
			im_size,
			task_filter,
		)
		return

	single_checkpoint = cli_args.checkpoint_path
	if single_checkpoint:
		ckpt_path = single_checkpoint.expanduser()
		if not ckpt_path.is_file():
			raise SystemExit(f"Checkpoint not found: {ckpt_path}")
		output_path = (
			cli_args.per_checkpoint_output.expanduser()
			if cli_args.per_checkpoint_output
			else tables_dir / f"{ckpt_path.stem}.csv"
		)
		existing_row: dict[str, str] | None = None
		if output_path.exists():
			try:
				with output_path.open("r", newline="") as handle:
					reader = csv.DictReader(handle)
					existing_row = next(reader, None)
			except Exception as exc:
				print(f"Warning: failed to parse existing CSV {output_path}: {exc}")
		if cli_args.force_recompute:
			existing_row = None
		model = build_conditional_ddim(
			in_channel=channels,
			image_size=im_size,
			num_class_labels=cli_args.num_classes,
			ewc_lambda=0.0,
			gr_kl=0.0,
		).to(device)
		state = torch.load(ckpt_path, map_location=device)
		model.load_state_dict(state)
		label = cli_args.checkpoint_label
		if not label:
			label, _ = _extract_checkpoint_label(ckpt_path)
		if cli_args.num_eval_tasks is not None:
			eval_task_ids = fixed_eval_task_ids(cli_args.num_eval_tasks, len(cl_test_loader))
		elif cli_args.task_id is not None:
			eval_task_ids = build_eval_task_ids(
				cli_args.task_id, len(cl_test_loader), cli_args.previous_only
			)
		else:
			raise SystemExit(
				"Provide --num_eval_tasks or --task_id when evaluating a single checkpoint."
			)
		if not eval_task_ids:
			print("No evaluation tasks configured; exiting.")
			return
		missing_task_ids = [
			tid
			for tid in eval_task_ids
			if not existing_row or not str(existing_row.get(f"task{tid}", "")).strip()
		]
		if not missing_task_ids:
			print(f"All requested tasks already evaluated for {ckpt_path}; skipping.")
			return
		updated_row: dict[str, str] = {"checkpoint": label}
		if existing_row:
			for key, value in existing_row.items():
				if not value:
					continue
				if key == "checkpoint" and not updated_row.get("checkpoint"):
					updated_row["checkpoint"] = value
				else:
					updated_row[key] = value
		for eval_task_id in missing_task_ids:
			fid = evaluate_fid(
				model,
				cl_test_loader[eval_task_id],
				device,
				num_inference_steps=cli_args.num_inference_steps,
				seed=cli_args.seed,
				max_real=cli_args.max_real,
			)
			col = f"task{eval_task_id}"
			updated_row[col] = f"{fid:.6f}"
			print(f"checkpoint {label} vs task {eval_task_id}: FID {fid:.3f}")
		fieldnames = ["checkpoint"] + [f"task{i}" for i in eval_task_ids]
		write_csv(output_path, fieldnames, [updated_row])
		print(f"Wrote checkpoint table: {output_path}")
		return

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

	summary_entries: list[tuple[str, dict[int, float]]] = []

	for run_dir in run_dirs:
		run_name = run_dir.name
		print(f"Evaluating run: {run_name}")
		model_tables_dir = tables_dir / run_name
		model_tables_dir.mkdir(parents=True, exist_ok=True)

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

		fid_csv = tables_dir / f"{run_name}_fid_by_task.csv"
		if cli_args.force_recompute:
			per_task_fids: dict[int, dict[int, float]] = {}
		else:
			per_task_fids = read_task_matrix_csv(fid_csv)
		updated_any = False

		for task_id, ckpt_path in task_ckpts:
			if cli_args.num_eval_tasks is not None:
				eval_task_ids = fixed_eval_task_ids(cli_args.num_eval_tasks, len(cl_test_loader))
			else:
				eval_task_ids = build_eval_task_ids(
					task_id, len(cl_test_loader), cli_args.previous_only
				)
			existing_results = {} if cli_args.force_recompute else dict(per_task_fids.get(task_id, {}))
			if task_id not in per_task_fids:
				per_task_fids[task_id] = existing_results
			missing_eval_ids = eval_task_ids if cli_args.force_recompute else [
				tid for tid in eval_task_ids if tid not in existing_results
			]
			if missing_eval_ids:
				print(f"	Loading checkpoint: {ckpt_path.name}")
				state = torch.load(ckpt_path, map_location=device)
				model.load_state_dict(state)
				for eval_task_id in missing_eval_ids:
					fid = evaluate_fid(
						model,
						cl_test_loader[eval_task_id],
						device,
						num_inference_steps=cli_args.num_inference_steps,
						seed=cli_args.seed,
						max_real=cli_args.max_real,
					)
					existing_results[eval_task_id] = fid
					print(f"		Task {task_id} vs Task {eval_task_id}: FID {fid:.3f}")
				per_task_fids[task_id] = existing_results
				completed_vals = [existing_results[idx] for idx in existing_results]
				if completed_vals:
					avg_fid = float(sum(completed_vals) / len(completed_vals))
					print(f"		Avg FID after task {task_id}: {avg_fid:.3f}")
				write_task_matrix_csv(fid_csv, per_task_fids, column_prefix="")
				updated_any = True
			else:
				print(
					f"	[resume] Task {task_id} already has requested evaluation tasks; skipping re-run."
				)
				if existing_results:
					completed_vals = [existing_results[idx] for idx in existing_results]
					avg_fid = float(sum(completed_vals) / len(completed_vals))
					print(f"		Avg FID after task {task_id}: {avg_fid:.3f}")

			checkpoint_entries = list_task_checkpoint_series(run_dir, task_id)
			if checkpoint_entries:
				checkpoint_csv = model_tables_dir / f"task{task_id}_checkpoint_fids.csv"
				wrote_checkpoint = build_checkpoint_rows(
					checkpoint_entries,
					model,
					cl_test_loader,
					eval_task_ids,
					device,
					cli_args.num_inference_steps,
					cli_args.seed,
					cli_args.max_real,
					checkpoint_csv,
					cli_args.force_recompute,
				)
				if wrote_checkpoint:
					print(f"		Updated checkpoint table: {checkpoint_csv}")
				else:
					print(f"		Checkpoint table already complete: {checkpoint_csv}")
			else:
				print(f"		No intermediate checkpoints found for task {task_id}; skipping table.")

		if per_task_fids:
			write_task_matrix_csv(fid_csv, per_task_fids, column_prefix="")
			if updated_any:
				print(f"Wrote task x task matrix: {fid_csv}")
			else:
				print(f"Task matrix already up to date: {fid_csv}")
		else:
			write_task_matrix_csv(fid_csv, {}, column_prefix="")
			print(f"No checkpoints evaluated for {run_name}; wrote empty matrix: {fid_csv}")

		avg_by_task: dict[int, float] = {}
		for task_id, results in per_task_fids.items():
			if results:
				avg_by_task[task_id] = float(sum(results.values()) / len(results))

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
