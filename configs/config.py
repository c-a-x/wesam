from box import Box

from configs.base_config import base_config


# Dataset-specific runners override these defaults. Keeping this configuration
# local makes importing the training modules independent of external projects.
config = {
    "gpu_ids": "0",
    # BF16 halves most Transformer activation storage on the target Blackwell
    # GPUs while keeping model weights and loss accumulation in FP32.
    "precision": "bf16-mixed",
    "batch_size": 4,
    "val_batchsize": 16,
    "num_workers": 0,
    "num_epochs": 10,
    "max_nums": 50,
    "resume": False,
    "dataset": "ISIC",
    "visual": False,
    "load_type": "soft",
    "prompt": "point",
    "out_dir": "./outputs",
    "name": "ifp_wesam_semisup",
    "semi": {
        "enabled": True,
        "labeled_ratio": 0.1,
        "labeled_count": None,
        "seed": 1337,
        "labeled_batch_probability": 0.5,
    },
    "teacher_weight": 0.1,
    "anchor_weight": 0.0,
    "contrast_weight": 0.0,
    "supervised_weight": 1.0,
    "unsupervised_rampup_epochs": 3,
    "model": {
        "multimask_output": False,
    },
    "prompt_generator": {
        "backend": "ifp",
        "ifp_root": "./assets",
        "alignment_checkpoint": "",
        "dino_variant": "dino3h",
        "clip_variant": "clipl",
        "text_prompts": [],
        "max_positive_points": 1,
        "min_positive_points": 1,
        "include_box": False,
        "labeled_prompt_mode": "gt",
        "iterative_pseudo": {
            "enabled": True,
            "max_iters": 1,
            "teacher_iou_threshold": 0.8,
            "min_new_pixels": 32,
            "min_new_area_ratio": 0.001,
            "max_mask_area_ratio": 0.6,
            "min_overlap_ratio": 0.7,
            "max_new_area_ratio": 0.4,
            "min_student_teacher_iou": 0.0,
        },
    },
}

cfg = Box(base_config)
cfg.merge_update(config)
