import os
import cv2
import numpy as np
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.tools import create_csv, write_csv
from utils.prompt_policy import apply_prompt_policy
from PIL import Image

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_prompt_on_image(image_np, prompt, mask=None, padding=None, save_path=None):
    """
    image_np: np.ndarray (H, W, 3), uint8, image with padding removed
    prompt: dict with 'in_points' and optional 'in_box' (torch tensors)
    mask: np.ndarray (H, W), optional, binary or uint8
    padding: (left, top, right, bottom), coordinates are adjusted if given
    save_path: optional path to save image
    """
    image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.copy()
    img_vis = image_np.copy()

    # Step 1: draw mask
    if mask is not None:
        mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0).astype(np.uint8)
        color_mask = np.zeros_like(img_vis)
        color_mask[:, :, 2] = 255  # green channel
        alpha = 0.5
        img_vis = np.where(mask[..., None], (1 - alpha) * img_vis + alpha * color_mask, img_vis).astype(np.uint8)

    draw_point =  "point" in cfg.prompt
    draw_box = "box" in cfg.prompt

    # Step 2: draw points
    if draw_point:
        input_point, input_label = prompt["in_points"]
        input_point = input_point[0].cpu().clone().numpy()
        input_label = input_label[0].cpu().numpy()

        if padding is not None:
            pad_left, pad_top, _, _ = padding
            input_point -= np.array([pad_left, pad_top])

        for pt, label in zip(input_point.astype(int), input_label):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(img_vis, tuple(pt), radius=5, color=color, thickness=-1)

    # Step 3: draw box
    if draw_box and prompt.get("in_box") is not None and prompt["in_box"] is not None:
        box = prompt["in_box"][0].cpu().clone().numpy().astype(int)
        if padding is not None:
            pad_left, pad_top, _, _ = padding
            box[[0, 2]] -= pad_left  # x1, x2
            box[[1, 3]] -= pad_top   # y1, y2
        x1, y1, x2, y2 = box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)

    # Step 4: save
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    # SAM2 returns mask logits. Threshold after sigmoid so this agrees with
    # the binary masks used by the focal and Dice losses.
    pred_mask = (torch.sigmoid(pred_mask) >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    # if cfg.prompt == "box" or cfg.prompt == "coarse":
    #     prompts = bboxes
    # elif cfg.prompt == "point":
    #     prompts = get_point_prompts(gt_masks, bboxes, cfg.num_points)
    # else:
    #     raise ValueError("Prompt Type Error!")
    prompts = get_point_prompts(gt_masks, bboxes, cfg.num_points)

    return prompts

def remove_padding(tensor, padding):
    """
    从一个 tensor 中移除指定 padding 区域。
    tensor: [C, H, W] 或 [H, W]
    padding: (left, top, right, bottom)
    """
    pad_left, pad_top, pad_right, pad_bottom = padding
    if tensor.ndim == 3:
        return tensor[:, pad_top: -pad_bottom if pad_bottom > 0 else None,
                      pad_left: -pad_right if pad_right > 0 else None]
    else:
        return tensor[pad_top: -pad_bottom if pad_bottom > 0 else None,
                      pad_left: -pad_right if pad_right > 0 else None]


def combine_instance_masks(mask):
    """Convert per-instance masks to the single foreground mask used by SAM2."""
    if mask.dim() == 3:
        return (mask.sum(dim=0, keepdim=True) > 0).float()
    return (mask > 0).float()


def validate(fabric: L.Fabric, cfg: Box, model: Model, dino, val_dataloader: DataLoader, image_prompt_path, mask_prompt_path, name: str, epoch: int = 0, save_mask: bool = False, prompt_generator=None):
    was_training = model.training
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    print("cfg.visual:", cfg.visual)


    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            data = list(data)
            if cfg.visual:
                # data = list(data)
                for idx in range(len(data[0])):  # data[0] 是 image_name 列表
                    image_name = data[0][idx]
                    padding = data[1][idx]
                    image = data[5][idx].unsqueeze(0).to(fabric.device)
                    bboxes = data[6][idx].unsqueeze(0).to(fabric.device)
                    gt_masks = data[7][idx].unsqueeze(0).to(fabric.device)
                    # _, new_H, new_W = image.shape
                    # print(image.shape)

                    prompts = (prompt_generator(image, [gt_masks.squeeze(0)])
                               if getattr(prompt_generator, "requires_gt_masks", False)
                               else prompt_generator(image))
                    prompts = apply_prompt_policy(
                        prompts, cfg.prompt_generator, image.shape[-2:], training=False
                    )
                    _, pred_masks, _, _ = model(image, prompts)

                    # ----------- Step 1: 提取最大轮廓 & 计算指标（原尺寸） -------------
                    pred_mask = pred_masks[0]  # shape: [1, H, W]
                    gt_mask = combine_instance_masks(gt_masks[0])

                    pred_mask_np = pred_mask.squeeze().detach().cpu().numpy()
                    pred_mask_tensor = pred_mask.squeeze().detach().cpu().unsqueeze(0).float().to(fabric.device)  # ← 新增 tensor 版本
                    binary_mask = (pred_mask_np > 0.5).astype(np.uint8) * 255

                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        max_contour_mask = np.zeros_like(binary_mask)
                        cv2.drawContours(max_contour_mask, [max_contour], -1, color=255, thickness=-1)
                    else:
                        max_contour_mask = np.zeros_like(binary_mask)

                    max_contour_tensor = torch.tensor(max_contour_mask / 255.0).unsqueeze(0).float().to(fabric.device)

                    # ----------- Step 2: 计算指标（注意：gt_mask 是原尺寸） -------------
                    batch_stats = smp.metrics.get_stats(
                        max_contour_tensor,
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

                    # ----------- Step 3: 去 padding，准备可视化 & 保存 -------------
                    image_np = remove_padding(image.squeeze(0).cpu(), padding).permute(1, 2, 0).numpy()
                    pred_mask_np = remove_padding(pred_mask.squeeze(0).cpu(), padding).numpy()
                    max_contour_mask_np = remove_padding(torch.tensor(max_contour_mask), padding).numpy()

                    # ----------- Step 4: 保存去 padding 的 mask 图 -------------
                    save_dir = os.path.join(cfg.out_dir, "pred_masks")
                    os.makedirs(save_dir, exist_ok=True)
                    pred_binary_mask = (pred_mask_np > 0.5).astype(np.uint8) * 255
                    Image.fromarray(pred_binary_mask).save(os.path.join(save_dir, f"{image_name}.png"))

                    save_mask_dir = os.path.join(cfg.out_dir, "max_pred_masks")
                    os.makedirs(save_mask_dir, exist_ok=True)
                    Image.fromarray(max_contour_mask_np.astype(np.uint8)).save(os.path.join(save_mask_dir, f"{image_name}.png"))

                    # ----------- Step 5: 可视化 -------------
                    save_img_dir = os.path.join(cfg.out_dir, "img_vis")
                    os.makedirs(save_img_dir, exist_ok=True)
                    vis_path = os.path.join(save_img_dir, f"{image_name}_vis.png")
                    visualize_prompt_on_image(image_np, prompts[0], mask=max_contour_mask_np, padding=padding, save_path=vis_path)
                    print(f"Saved visualization to {vis_path}")


                    # gt_mask_img = Image.fromarray(gt_mask.squeeze().cpu().numpy().astype('uint8') * 255)
                    # gt_mask_img.save(os.path.join(save_dir, f"{image_name}_gt.png"))
            else:
                # print(f"[DEBUG] data type: {type(data)}, content type: {[type(i) for i in data]}")
                # data = list(data)
                images, bboxes, gt_masks = data
                num_images = images.size(0)
                # print('num_images',num_images)
                # prompts = get_prompts(cfg, bboxes, gt_masks)
                # print('prompts',prompts)

                prompts = (prompt_generator(images, gt_masks)
                           if getattr(prompt_generator, "requires_gt_masks", False)
                           else prompt_generator(images))
                prompts = apply_prompt_policy(
                    prompts, cfg.prompt_generator, images.shape[-2:], training=False
                )

                _, pred_masks, _, _ = model(images, prompts)
                for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                    gt_mask = combine_instance_masks(gt_mask)
                    batch_stats = smp.metrics.get_stats(
                        pred_mask,
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    # Metrics are computed one image at a time, so every image
                    # must contribute exactly once, including a partial last batch.
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_path = os.path.join(cfg.out_dir, "metrics.csv")
    csv_dict = {"Dataset": cfg.dataset, "Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "epoch": epoch}

    if fabric.global_rank == 0:
        create_csv(csv_path, csv_head=cfg.csv_keys)
        write_csv(csv_path, csv_dict, csv_head=cfg.csv_keys)
    model.train(was_training)
    return ious.avg, f1_scores.avg
