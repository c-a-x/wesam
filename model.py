import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam_lora import LoRA_Sam2
from configs.config import cfg


class Model(nn.Module):
    mask_threshold: float = 0.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None
        self.vision_pos_enc = None
        self.high_res_feats = None

    def setup(self):
        self.model = build_sam2(
            config_file=self.cfg.model.sam2_config,
            ckpt_path=self.cfg.model.ckpt,
            device="cpu",  # fabric will move to correct device
            mode="train",
        )

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = False

        self.finetune()

    def finetune(self):
        LoRA_Sam2(self.model, 4)

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    @property
    def image_size(self):
        return self.model.image_size

    def forward(self, images, prompts=None):
        _, _, H, W = images.shape
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode((H, W), prompts)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        """Encode images through SAM2 backbone + FPN neck.

        Returns:
            image_embeddings: (B, 256, H/16, W/16) vision features
        """
        backbone_out = self.model.forward_image(images)

        # Extract high-res features for mask decoder
        if self.model.use_high_res_features_in_sam:
            # forward_image already applies conv_s0/conv_s1
            pass

        # Prepare backbone features
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.model._prepare_backbone_features(backbone_out)

        # Reshape flattened feats back to spatial format
        # vision_feats[-1] is (HW, B, C) → (B, C, H, W)
        feats = [
            feat.permute(1, 2, 0).view(feat.shape[1], -1, *fsize)
            for feat, fsize in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        self.image_embeddings = feats[-1]  # main embedding
        self.vision_pos_enc = backbone_out["vision_pos_enc"]
        self.high_res_feats = feats[:-1]  # high-res features for decoder

        return self.image_embeddings

    def decode(self, image_shape, prompts=None):
        """Decode masks from image embeddings and prompts.

        Args:
            image_shape: (H, W) original image size
            prompts: optional list of dicts with 'in_points' and/or 'in_box'.
                When None, SAM2 receives no point, box, or mask prompt.

        Returns:
            pred_masks, ious, res_masks
        """
        if self.image_embeddings is None:
            raise RuntimeError("No image embeddings. Call encode() first.")
        if prompts is None:
            prompts = [None] * len(self.image_embeddings)
        if len(prompts) != len(self.image_embeddings):
            raise ValueError("Prompt count must match the encoded image batch size.")

        multimask_output = False  # Old fair WeSAM protocol uses SAM2 single-mask decoding.

        pred_masks = []
        ious = []
        res_masks = []

        for i, (prompt, embedding) in enumerate(zip(prompts, self.image_embeddings)):
            if prompt is None:
                in_points = None
                input_box = None
            else:
                in_points = prompt.get("in_points", None)
                input_box = prompt.get("in_box", None)

            # Move to model device
            if in_points is not None:
                in_points = (in_points[0].to(embedding.device),
                             in_points[1].to(embedding.device))
            if input_box is not None:
                input_box = input_box.to(embedding.device)

            # Handle box prompts: SAM2 encodes boxes as point pairs
            if cfg.prompt == "box" and input_box is not None:
                box_coords = input_box.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int,
                                          device=embedding.device)
                box_labels = box_labels.repeat(input_box.shape[0], 1)
                concat_points = (box_coords, box_labels)

            elif cfg.prompt == "none":
                concat_points = None

            elif cfg.prompt == "point":
                concat_points = in_points

            elif cfg.prompt in ["point+box", "point+Box"]:
                if input_box is not None:
                    box_coords = input_box.reshape(-1, 2, 2)
                    box_labels = torch.tensor([[2, 3]], dtype=torch.int,
                                              device=embedding.device)
                    box_labels = box_labels.repeat(input_box.shape[0], 1)
                    if in_points is not None:
                        concat_coords = torch.cat([box_coords, in_points[0]], dim=1)
                        concat_labels = torch.cat([box_labels, in_points[1]], dim=1)
                        concat_points = (concat_coords, concat_labels)
                    else:
                        concat_points = (box_coords, box_labels)
                else:
                    concat_points = in_points
            else:
                raise ValueError(f"Unsupported prompt mode: {cfg.prompt}")

            # Encode prompts
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )

            # Prepare high-res features for this sample
            high_res_features = [
                feat_level[i].unsqueeze(0)
                for feat_level in self.high_res_feats
            ] if self.high_res_feats else None

            # Decode masks
            low_res_mask, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # IFP parity: when SAM emits several masks for a prompt, keep the
            # one SAM itself scores highest (argmax IoU prediction), exactly
            # like the original IFP SamPredictor(multimask_output=True) path.
            if multimask_output and low_res_mask.shape[1] > 1:
                best = int(torch.argmax(iou_predictions[0]).item())
                low_res_mask = low_res_mask[:, best:best + 1]
                iou_predictions = iou_predictions[:, best:best + 1]

            masks = F.interpolate(
                low_res_mask,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )

            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_mask)

        return pred_masks, ious, res_masks
