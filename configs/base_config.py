base_config = {
    "eval_interval": 1,
    "skip_initial_validation": False,
    "ema_rate": 0.999,
    "semi": {
        "enabled": True,
        "labeled_ratio": 0.1,
        "seed": 1337,
        # With a small labeled subset, sample labeled examples more often so
        # supervised updates are not drowned out by pseudo-label updates.
        "labeled_batch_probability": 0.5,
        "labeled_batch_probability": 0.5,
    },
    "teacher_weight": 1.0,
    "anchor_weight": 1.0,
    "contrast_weight": 1.0,
    "supervised_weight": 1.0,   # GT 监督 loss 权重
    "unsupervised_rampup_epochs": 3,
    "csv_keys": ["Dataset", "Name", "Prompt", "Mean IoU", "Mean F1", "epoch"],
    "prompt_generator": {
        "backend": "ifp",
        "ifp_root": "./assets",
        "alignment_checkpoint": "",
        "background_alignment_checkpoint": "",
        "dino_variant": "dino3h",
        "clip_variant": "clipl",
        "text_prompts": [],
        "background_text_prompts": [],
        "resize": 0,
        "score_threshold": None,
        "max_positive_points": 1,
        "min_positive_points": 1,
        "include_box": False,
        "iterative_pseudo": {
            "enabled": False,
            "max_iters": 3,
            "teacher_iou_threshold": 0.5,
            "min_new_pixels": 32,
            "min_new_area_ratio": 0.001,
            "max_mask_area_ratio": 0.8,
            "min_overlap_ratio": 0.5,
            "max_new_area_ratio": 0.4,
        },
    },
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": "sam2.1_hiera_l",
        "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "ckpt": "./checkpoints/sam2.1_hiera_large.pt",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "ISIC": {
            "root_dir": "./Data/ISIC2018",
            "train_list": "./Data/ISIC2018/manifests/train.csv",
            "test_list": "./Data/ISIC2018/manifests/validation.csv",
        },
        "Kvasir": {
            "root_dir": "./Data/kvasir-seg/organized/Kvasir",
            "train_list": "./Data/kvasir-seg/organized/Kvasir/train.csv",
            "test_list": "./Data/kvasir-seg/organized/Kvasir/validation.csv",
        },
        "Polyp": {
            "root_dir": "./Data/kvasir-seg/organized",
            "train_list": "",
            "test_list": "",
        },
    },
}
