# Adapted from Sheng Wang's LoRA_Sam for SAM 2

from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file


class _LoRA_qkv_proj(nn.Module):
    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.proj(x) + self.w_b(self.w_a(x))
        return x


class _LoRA_qkv(nn.Module):
    """LoRA wrapper for Hiera's MultiScaleAttention.qkv

    SAM 2 Hiera uses:
        self.qkv = nn.Linear(dim, dim_out * 3)
    where dim and dim_out can differ at stage boundaries.
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.dim_out = qkv.out_features // 3

    def forward(self, x):
        # x: (B, H, W, C)
        qkv = self.qkv(x)  # (B, H, W, 3*dim_out)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # Q is the first dim_out channels, V is the last dim_out channels
        qkv[:, :, :, : self.dim_out] += new_q
        qkv[:, :, :, -self.dim_out :] += new_v
        return qkv


class LoRA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_fc_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class LoRA_Sam2(LoRA):
    """Applies low-rank adaptation to a SAM 2 model's Hiera image encoder.

    Args:
        sam2_model: a SAM2Base model
        r: rank of LoRA
        lora_layer: which block indices to apply LoRA (None = all)

    Key difference from SAM 1:
        SAM 1: model.image_encoder.blocks[i].attn.qkv
        SAM 2: model.image_encoder.trunk.blocks[i].attn.qkv
    """

    def __init__(self, sam2_model: SAM2Base, r: int, lora_layer=None):
        super(LoRA_Sam2, self).__init__()

        assert r > 0

        # Hiera trunk blocks
        trunk = sam2_model.image_encoder.trunk

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(trunk.blocks)))

        # create for storage, then we can init them or load weights
        self.w_As = []
        self.w_Bs = []

        # freeze all parameters first
        for param in sam2_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam2_model.sam_mask_decoder.parameters():
            param.requires_grad = False

        # LoRA surgery on Hiera trunk blocks
        for t_layer_i, blk in enumerate(trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            dim_out = w_qkv_linear.out_features // 3

            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim_out, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim_out, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.lora_vit = sam2_model
