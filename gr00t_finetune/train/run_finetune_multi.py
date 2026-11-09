#!/usr/bin/env python3
"""Launch GR00T N1.7 finetune on multiple datasets (mirrors launch_finetune.py but takes N paths).

Splits a colon-separated `--dataset_path` into N datasets, each with mix_ratio=1.0 so the factory
balances by dataset length. Rest of the config mirrors examples/finetune.sh defaults.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro


@dataclass
class MultiFinetuneConfig:
    base_model_path: str
    dataset_path: str  # colon-separated list of dataset roots
    embodiment_tag: str
    output_dir: str
    experiment_name: str
    modality_config_path: str | None = None
    max_steps: int = 20000
    save_steps: int = 2000
    save_total_limit: int = 1
    global_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    num_gpus: int = 1
    dataloader_num_workers: int = 4
    shard_size: int = 1024
    num_shards_per_epoch: int = 100000
    episode_sampling_rate: float = 0.1
    use_wandb: bool = False
    wandb_project: str = ""
    state_dropout_prob: float = 0.2
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    color_jitter_params: dict[str, float] | None = field(
        default_factory=lambda: {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    )
    random_rotation_angle: int | None = None
    extra_augmentation_config: str | None = None
    save_only_model: bool = False
    skip_weight_loading: bool = False


def load_modality_config(path_str: str) -> None:
    import importlib
    p = Path(path_str)
    if not (p.exists() and p.suffix == ".py"):
        raise FileNotFoundError(p)
    sys.path.append(str(p.parent))
    importlib.import_module(p.stem)
    print(f"Loaded modality config: {p}")


def main() -> None:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    cfg = tyro.cli(MultiFinetuneConfig)

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    tag = EmbodimentTag.resolve(cfg.embodiment_tag).value

    if cfg.modality_config_path:
        load_modality_config(cfg.modality_config_path)

    dataset_paths = [p for p in cfg.dataset_path.split(":") if p]
    assert dataset_paths, "no dataset paths provided"
    for p in dataset_paths:
        assert Path(p).is_dir(), f"not a dir: {p}"

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = cfg.tune_llm
    config.model.tune_visual = cfg.tune_visual
    config.model.tune_projector = cfg.tune_projector
    config.model.tune_diffusion_model = cfg.tune_diffusion_model
    config.model.state_dropout_prob = cfg.state_dropout_prob
    config.model.random_rotation_angle = cfg.random_rotation_angle
    config.model.color_jitter_params = cfg.color_jitter_params
    config.model.extra_augmentation_config = (
        json.loads(cfg.extra_augmentation_config) if cfg.extra_augmentation_config else None
    )
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = cfg.experiment_name
    config.training.start_from_checkpoint = cfg.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = cfg.global_batch_size
    config.training.dataloader_num_workers = cfg.dataloader_num_workers
    config.training.learning_rate = cfg.learning_rate
    config.training.gradient_accumulation_steps = cfg.gradient_accumulation_steps
    config.training.output_dir = cfg.output_dir
    config.training.save_steps = cfg.save_steps
    config.training.save_total_limit = cfg.save_total_limit
    config.training.num_gpus = cfg.num_gpus
    config.training.use_wandb = cfg.use_wandb
    config.training.max_steps = cfg.max_steps
    config.training.weight_decay = cfg.weight_decay
    config.training.warmup_ratio = cfg.warmup_ratio
    config.training.wandb_project = cfg.wandb_project
    config.training.save_only_model = cfg.save_only_model
    config.training.skip_weight_loading = cfg.skip_weight_loading

    config.data.shard_size = cfg.shard_size
    config.data.episode_sampling_rate = cfg.episode_sampling_rate
    config.data.num_shards_per_epoch = cfg.num_shards_per_epoch

    run(config)


if __name__ == "__main__":
    main()
