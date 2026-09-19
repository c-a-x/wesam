import os
import cv2
import random
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from skimage.draw import polygon2mask

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft, collate_fn_ifp_oracle, collate_fn_, decode_mask


class ISICDataset(Dataset):
    def __init__(self, cfg, root_dir, list_file, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        df = pd.read_csv(os.path.join(list_file), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

        self.if_self_training = if_self_training
        self.has_gt_list = self._build_has_gt_list()
        oracle_cfg = getattr(cfg, "ifp_oracle", {})
        self.oracle_cache = None
        if bool(getattr(oracle_cfg, "enabled", False)):
            cache_path = getattr(oracle_cfg, "cache_path", "")
            if not cache_path or not os.path.isfile(cache_path):
                raise FileNotFoundError(f"IFP oracle cache not found: {cache_path}")
            self.oracle_cache = torch.load(cache_path, map_location="cpu", weights_only=False)["entries"]

    def _build_has_gt_list(self):
        semi_cfg = getattr(self.cfg, "semi", {})
        enabled = bool(getattr(semi_cfg, "enabled", False))
        ratio = float(getattr(semi_cfg, "labeled_ratio", 1.0))
        labeled_count = getattr(semi_cfg, "labeled_count", None)
        explicit_indices = getattr(semi_cfg, "labeled_indices", None)
        seed = int(getattr(semi_cfg, "seed", 1337))

        if not self.training or not enabled:
            return [True] * len(self.name_list)

        if explicit_indices is not None:
            labeled_indices = {int(index) for index in explicit_indices}
            if any(index < 0 or index >= len(self.name_list) for index in labeled_indices):
                raise ValueError("semi.labeled_indices contains an out-of-range index.")
            return [idx in labeled_indices for idx in range(len(self.name_list))]
        if labeled_count is None:
            ratio = min(max(ratio, 0.0), 1.0)
            num_labeled = round(len(self.name_list) * ratio)
        else:
            num_labeled = int(labeled_count)
            if not 0 <= num_labeled <= len(self.name_list):
                raise ValueError(
                    f"labeled_count must be within [0, {len(self.name_list)}], got {num_labeled}."
                )
        rng = random.Random(seed)
        labeled_indices = set(rng.sample(range(len(self.name_list)), num_labeled))
        return [idx in labeled_indices for idx in range(len(self.name_list))]

    def __len__(self):
        return len(self.name_list)

    def _oracle_prompt_and_mask(self, name, image):
        entry = self.oracle_cache.get(str(name).replace("\\", "/"))
        if entry is None:
            raise KeyError(f"No IFP oracle entry for {name}")
        points = entry["points"].clone().float()
        height, width = image.shape[:2]
        resized = self.transform.transform.apply_image(image)
        resized_height, resized_width = resized.shape[:2]
        pad_left = (max(resized_width, resized_height) - resized_width) // 2
        pad_top = (max(resized_width, resized_height) - resized_height) // 2
        points = self.transform.transform.apply_coords(points.unsqueeze(0), (height, width)).squeeze(0)
        points[:, 0] += pad_left
        points[:, 1] += pad_top
        if "mask_path" in entry:
            pseudo_source = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        else:
            pseudo_source = entry["mask"].numpy()
        pseudo = self.transform.transform.apply_image(pseudo_source)
        pad_right = max(resized_width, resized_height) - resized_width - pad_left
        pad_bottom = max(resized_width, resized_height) - resized_height - pad_top
        pseudo = torch.from_numpy(pseudo).float()
        pseudo = torch.nn.functional.pad(pseudo, (pad_left, pad_right, pad_top, pad_bottom))
        prompt = {
            "in_points": (
                points.unsqueeze(0),
                torch.ones((1, points.shape[0]), dtype=torch.int),
            ),
            "in_box": None,
        }
        return prompt, (pseudo > 127).float()

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)
            bboxes.append([x, y, x + w, y + h])
            categories.append("0")

        if self.oracle_cache is not None:
            prompt, pseudo_mask = self._oracle_prompt_and_mask(name, image)
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            if self.if_self_training:
                # Prompts/masks are cached in the same coordinate system, so keep both views identical.
                return (
                    image, image.clone(), torch.tensor(bboxes), torch.tensor(masks).float(),
                    self.has_gt_list[idx], prompt, pseudo_mask,
                )
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), prompt

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float(), self.has_gt_list[idx]

        elif self.cfg.visual:
            file_name = os.path.splitext(os.path.basename(name))[0]
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return file_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class ISICDatasetwithCoarse(ISICDataset):

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        approxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)

            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)
            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])

            masks.append(mask)
            categories.append("0")
            approxes.append(approx)

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float(), self.has_gt_list[idx]

        elif self.cfg.visual:
            file_name = os.path.splitext(os.path.basename(name))[0]
            origin_image = image
            origin_approxes = approxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), self.cfg.visual)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return file_name, padding, origin_image, origin_approxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    train = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        training=True,
    )
    oracle = bool(getattr(getattr(cfg, "ifp_oracle", {}), "enabled", False))
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_ifp_oracle if oracle else collate_fn,
    )
    train_sampler = _make_semi_supervised_sampler(train, cfg)
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_ifp_oracle if oracle else collate_fn,
    )
    return train_dataloader, val_dataloader


def _make_semi_supervised_sampler(dataset, cfg):
    """Balance labeled and unlabeled update frequency for semi-supervision."""
    semi_cfg = getattr(cfg, "semi", {})
    if not bool(getattr(semi_cfg, "enabled", False)):
        return None

    labels = torch.tensor(dataset.has_gt_list, dtype=torch.bool)
    labeled = int(labels.sum().item())
    unlabeled = int((~labels).sum().item())
    if labeled == 0 or unlabeled == 0:
        return None

    labeled_probability = float(
        getattr(semi_cfg, "labeled_batch_probability", 0.5)
    )
    labeled_probability = min(max(labeled_probability, 0.0), 1.0)

    weights = torch.where(
        labels,
        torch.full_like(labels, labeled_probability / labeled, dtype=torch.float),
        torch.full_like(labels, (1.0 - labeled_probability) / unlabeled, dtype=torch.float),
    )
    generator = torch.Generator().manual_seed(int(getattr(semi_cfg, "seed", 1337)))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    soft_train = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    oracle = bool(getattr(getattr(cfg, "ifp_oracle", {}), "enabled", False))
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_ifp_oracle if oracle else collate_fn,
    )
    train_sampler = _make_semi_supervised_sampler(soft_train, cfg)
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_ifp_oracle if oracle else collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    train = ISICDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_soft_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    soft_train = ISICDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.ISIC.root_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader
