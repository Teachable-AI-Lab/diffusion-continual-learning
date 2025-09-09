import torch
from src.ddim import build_conditional_ddim


def count_params(model: torch.nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(variant: str, in_channels: int, image_size: int, num_classes: int):
	if variant == "small":
		return build_conditional_ddim(
			in_channel=in_channels,
			image_size=image_size,
			num_class_labels=num_classes,
			block_out_channels=(16,),
			down_block_types=("DownBlock2D",),
			up_block_types=("UpBlock2D",),
			norm_num_groups=8,
			layers_per_block=1,
		)
	elif variant == "small-double":
		return build_conditional_ddim(
			in_channel=in_channels,
			image_size=image_size,
			num_class_labels=num_classes,
			block_out_channels=(16, 16, 16),
			down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
			up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
			norm_num_groups=8,
			layers_per_block=2,
		)
	elif variant == "small-big":
		return build_conditional_ddim(
			in_channel=in_channels,
			image_size=image_size,
			num_class_labels=num_classes,
			block_out_channels=(16, 32, 16),
			down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
			up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
			norm_num_groups=8,
			layers_per_block=1,
		)
	elif variant == "big":
		# Use defaults from src.ddim based on image_size
		return build_conditional_ddim(
			in_channel=in_channels,
			image_size=image_size,
			num_class_labels=num_classes,
		)
	else:
		raise ValueError(f"Unknown variant: {variant}")


def main():
	# Defaults matching common setups in this repo
	print("Model parameter counts for various DDIM variants:\n")
	image_size = 32
	num_classes = 10
	variants = ["small", "small-double", "small-big", "big"]
	channel_setups = [
		(1, "grayscale"),
		(3, "rgb"),
	]

	for in_ch, name in channel_setups:
		print(f"\n== {name.upper()} (C={in_ch}, H=W={image_size}, classes={num_classes}) ==")
		for v in variants:
			model = make_model(v, in_ch, image_size, num_classes)
			n = count_params(model)
			print(f"{v:12s}: {n:,} params")


if __name__ == "__main__":
	main()

