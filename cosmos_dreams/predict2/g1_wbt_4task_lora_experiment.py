from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

g1_wbt_4task_dataset = L(Dataset)(
    dataset_dir="/workspace/data/cosmos_train/g1_wbt_2task",
    num_frames=93,
    video_size=(432, 768),
)

dataloader_train_g1_wbt_4task = L(DataLoader)(
    dataset=g1_wbt_4task_dataset,
    sampler=L(get_sampler)(dataset=g1_wbt_4task_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Resumes from 2task iter_000004000.pt, adds washing + fridge (drinks) data,
# runs 4000 more iters for total 8000.
predict2_video2world_lora_training_2b_g1_wbt_4task = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_g1_wbt_4task",
    ),
    model=dict(
        config=dict(
            train_architecture="lora",
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=False,
            fsdp_shard_size=1,
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_g1_wbt_4task,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=8000,
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path="/workspace/runs/posttraining/video2world_lora/2b_g1_wbt_2task/checkpoints/model/iter_000004000.pt",
        load_training_state=False,
        strict_resume=False,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[200],
        cycle_lengths=[4000],
        f_max=[0.6],
        f_min=[0.0],
    ),
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_lora_training_2b_g1_wbt_4task",
    node=predict2_video2world_lora_training_2b_g1_wbt_4task,
)
