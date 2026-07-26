import os
import cv2
import torch
# import kornia as K
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from imagecorruptions import corrupt, get_corruption_names

weak_transforms = A.Compose(
    [A.HorizontalFlip(), A.VerticalFlip()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    # keypoint_params=A.KeypointParams(format='xy')
)

strong_transforms = A.Compose(
    [
        A.Posterize(),
        A.Equalize(),
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(),
        A.RandomShadow(),
    ]
)


def corrupt_image(image, filename):
    file_name = os.path.basename(os.path.abspath(filename))
    file_path = os.path.dirname(os.path.abspath(filename))
    for corruption in get_corruption_names():
        corrupted = corrupt(image, severity=5, corruption_name=corruption)
        corrupt_path = file_path.replace(
            "val2017", os.path.join("corruption", corruption)
        )
        if not os.path.exists(corrupt_path):
            os.makedirs(corrupt_path, exist_ok=True)
        cv2.imwrite(os.path.join(corrupt_path, file_name), corrupted)
# import cv2
# import albumentations as A

# # 假设这是你的图像路径
# image_path = '/data/gauss/wr/dataset/LXJ/LXJ1/train/image/2023_01_07_7-000104.jpg'

# # 加载图像
# image = cv2.imread(image_path)

# # 定义弱增强和强增强
# weak_transforms = A.Compose([A.Flip(), A.HorizontalFlip(), A.VerticalFlip()])
# strong_transforms = A.Compose([
#     A.Posterize(),
#     A.Equalize(),
#     A.Sharpen(),
#     A.Solarize(),
#     A.RandomBrightnessContrast(),
#     A.RandomShadow(),
# ])

# # 应用弱增强
# weak_augmented_image = weak_transforms(image=image)["image"]

# # 应用强增强
# strong_augmented_image = strong_transforms(image=image)["image"]

# # 保存增强后的图像
# cv2.imwrite('weak_augmented_image_2.jpg', weak_augmented_image)
# cv2.imwrite('strong_augmented_image_2.jpg', strong_augmented_image)
