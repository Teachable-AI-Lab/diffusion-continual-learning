"""
Configuration settings for diffusion continual learning experiments.
"""

import torch
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration class for experiments"""
    dataset: str = "mnist"
    epochs: int = 200
    batch_size: int = 128
    device: str = "auto"
    output_dir: str = "./experiment_results"
    seed: int = 123
    ewc_variants: List[str] = None
    run_full_fisher: bool = False
    learning_rate: float = 2e-4
    
    def __post_init__(self):
        # Set default EWC variants if not provided
        if self.ewc_variants is None:
            self.ewc_variants = ["diag", "rank1", "rank1_opt"]
        
        # Set device automatically
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        
        # Convert output directory to Path
        self.output_dir = Path(self.output_dir)
        
        # Model configuration based on dataset
        if self.dataset.lower() == "mnist":
            self.model_config = {
                "in_channel": 1,
                "image_size": 32,
                "num_class_labels": 4,  # 2 classes per task, 2 tasks
                "small_model_config": {
                    "block_out_channels": (16, 16),
                    "down_block_types": ("DownBlock2D", "DownBlock2D"),
                    "up_block_types": ("UpBlock2D", "UpBlock2D"),
                    "norm_num_groups": 8,
                    "layers_per_block": 1,
                }
            }
        elif self.dataset.lower() == "cifar10":
            self.model_config = {
                "in_channel": 3,
                "image_size": 32,
                "num_class_labels": 2,  # 5 classes per task, 2 tasks
                "small_model_config": {
                    "block_out_channels": (16, 16),
                    "down_block_types": ("DownBlock2D", "DownBlock2D"),
                    "up_block_types": ("UpBlock2D", "UpBlock2D"),
                    "norm_num_groups": 8,
                    "layers_per_block": 1,
                }
            }
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for saving/loading model"""
        return self.output_dir / f"{model_name}.pth"
    
    def get_results_path(self, filename: str) -> Path:
        """Get path for saving results"""
        return self.output_dir / filename
