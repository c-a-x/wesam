import random
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from datasets.augmentation import weak_transforms, strong_transforms


class ResizeLongestSide:
    """Resize image so the longest side equals target_size, preserving aspect ratio.

    Replaces segment_anything.utils.transforms.ResizeLongestSide (SAM1 dependency removed).
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def apply_image(self, image):
        """Resize image (HWC numpy array) so longest side = target_size."""
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        return cv2.resize(image, (new_w, new_h),
                          interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords, original_size):
        """Transform coordinates from original image space to resized space."""
        h, w = original_size
        scale = self.target_size / max(h, w)
        coords = coords.clone().float()
        coords[..., 0] *= scale
        coords[..., 1] *= scale
        return coords

    def apply_boxes(self, boxes, original_size):
        """Transform boxes from original image space to resized space."""
        boxes = np.array(boxes).reshape(-1, 2, 2)
        h, w = original_size
        scale = self.target_size / max(h, w)
        boxes = boxes.astype(np.float32) * scale
        return boxes.reshape(-1, 4)


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None, visual=False):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        if bboxes is not None:
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = [
                [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
                for bbox in bboxes
            ]
            if visual:
                return padding, image, masks, bboxes
            else:
                return image, masks, bboxes
        else:
            if visual:
                return padding, image, masks
            else:
                return image, masks

    def transform_image(self, image):
        # Resize image and masks
        image = self.transform.apply_image(image)
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        return image

    def transform_coord(self, points, image):
        og_h, og_w, _ = image.shape
        coords = points.reshape(1, -1, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, 2)

    def transform_coords(self, points, image, n):
        og_h, og_w, _ = image.shape
        coords = points.reshape(-1, n, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, n, 2)


def soft_transform(
    image: np.ndarray, bboxes: list, masks: list, categories: list
):
    weak_transformed = weak_transforms(
        image=image, bboxes=bboxes, masks=masks, category_ids=categories)
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, image_strong


def soft_transform_all(
    image: np.ndarray, bboxes: list, masks: list, points: list, categories: list
):
    weak_transformed = weak_transforms(
        image=image, bboxes=bboxes, masks=masks, category_ids=categories, keypoints=points)
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]
    keypoints_weak = weak_transformed["keypoints"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, keypoints_weak, image_strong


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


def collate_fn_soft(batch):
    images_soft, images, bboxes, masks, has_gts = zip(*batch)
    images = torch.stack(images)
    # images_origin = np.stack(images_origin)
    images_soft = torch.stack(images_soft)
    has_gts = torch.tensor(has_gts, dtype=torch.bool)
    return images_soft, images, bboxes, masks, has_gts


def collate_fn_ifp_oracle(batch):
    """Collate fixed IFP oracle prompts and masks without spatial augmentation."""
    if len(batch[0]) == 7:
        images_weak, images_strong, bboxes, masks, has_gts, prompts, pseudo_masks = zip(*batch)
        return (
            torch.stack(images_weak), torch.stack(images_strong), bboxes, masks,
            torch.tensor(has_gts, dtype=torch.bool), list(prompts), torch.stack(pseudo_masks),
        )
    images, bboxes, masks, prompts = zip(*batch)
    return torch.stack(images), bboxes, masks, list(prompts)


def collate_fn_coarse(batch):
    images, bboxes, masks, coarse_masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks, coarse_masks


def collate_fn_(batch):
    return zip(*batch)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


if __name__ == "__main__":
    mask_encode = np.array([[[0, 0, 1], [2, 0, 2], [0, 3, 3]]])
    mask_decode = np.array([[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 1, 1]]])
    encoded_mask = encode_mask(torch.tensor(mask_decode))
    decoded_mask = decode_mask(torch.tensor(mask_encode))
