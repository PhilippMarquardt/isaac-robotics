"""LoRA post-training config for cosmos-predict2.5, 2B Video2World.

Train set: pillow + clothes + washing (~3 tasks)
Val set:   fridge (held out for generalization eval)
Base ckpt: auto-downloaded via MODEL_CHECKPOINTS registry.
"""
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

g1_wbt_train = L(VideoDataset)(
    dataset_dir="datasets/g1_wbt/train",
    num_frames=93,
    video_size=(432, 768),
)
g1_wbt_val = L(VideoDataset)(
    dataset_dir="datasets/g1_wbt/val",
    num_frames=93,
    video_size=(432, 768),
)

dataloader_train_g1_wbt = L(get_generic_dataloader)(
    dataset=g1_wbt_train,
    sampler=L(get_sampler)(dataset=g1_wbt_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_g1_wbt = L(get_generic_dataloader)(
    dataset=g1_wbt_val,
    sampler=L(get_sampler)(dataset=g1_wbt_val),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

predict2p5_lora_training_2b_g1_wbt = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_g1_wbt_lora",
        wandb_mode="disabled",
    ),
    dataloader_train=dataloader_train_g1_wbt,
    dataloader_val=dataloader_val_g1_wbt,
    checkpoint=dict(
        load_path=DEFAULT_CHECKPOINT.s3.uri,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[500],
        cycle_lengths=[8000],
    ),
    trainer=dict(
        run_validation=True,
        validation_iter=500,
        logging_iter=50,
        max_iter=8000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=50, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            validation_draw_sample_reg=L(ValidationDrawSample)(
                n_samples=2, is_ema=False, save_s3=False, do_x0_prediction=False,
            ),
            validation_draw_sample_ema=L(ValidationDrawSample)(
                n_samples=2, is_ema=True, save_s3=False, do_x0_prediction=False,
            ),
        ),
    ),
    model=dict(
        config=dict(
            use_lora=True,
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            min_num_conditional_frames=0,
            max_num_conditional_frames=2,
            conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
            conditional_frame_timestep=-1.0,
            conditioning_strategy="frame_replace",
            denoise_replace_gt_frames=True,
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="predict2p5_lora_training_2b_g1_wbt",
    node=predict2p5_lora_training_2b_g1_wbt,
)
