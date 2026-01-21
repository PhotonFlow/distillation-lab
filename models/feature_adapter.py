from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from sam_mask_generator import SAMMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def _sample_uniform(
    shape: Sequence[int],
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (high - low) * torch.rand(*shape, device=device, dtype=dtype) + low


def _build_band_pass_mask(
    height: int,
    width: int,
    radius: float,
    band_width: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a (H, W) band-pass mask in the Fourier domain.

    radius and band_width are normalized to [0, 0.5] (Nyquist).
    If band_width is None or <= 0, a low-pass mask (<= radius) is used.
    """
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    radius = float(radius)
    if band_width is None or band_width <= 0:
        mask = dist <= radius
    else:
        half_bw = band_width * 0.5
        lower = max(0.0, radius - half_bw)
        upper = min(0.5, radius + half_bw)
        mask = (dist >= lower) & (dist <= upper)
    return mask.to(dtype=dtype)


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    return kernel2d


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    c, h, w = image.shape
    kernel_size = max(3, int(round(sigma * 3) * 2 + 1))
    kernel = _gaussian_kernel2d(kernel_size, sigma, image.device, image.dtype)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size)
    image_batched = image.unsqueeze(0)
    blurred = F.conv2d(image_batched, kernel, padding=kernel_size // 2, groups=c)
    return blurred.squeeze(0)


def _color_jitter(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    c, _, _ = image.shape
    out = image
    mean = out.mean(dim=(1, 2), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out * brightness
    if c >= 3:
        gray = out.mean(dim=0, keepdim=True)
        out = gray * (1.0 - saturation) + out * saturation
    return out


def _random_erasing(
    image: torch.Tensor,
    erase_scale: Tuple[float, float],
    erase_value: float = 0.0,
) -> torch.Tensor:
    c, h, w = image.shape
    area = h * w
    target_area = area * _sample_uniform((1,), erase_scale[0], erase_scale[1], image.device, image.dtype)
    target_area = float(target_area.item())
    if target_area <= 0:
        return image
    erase_h = int(round(math.sqrt(target_area)))
    erase_w = int(round(target_area / max(erase_h, 1)))
    erase_h = min(erase_h, h)
    erase_w = min(erase_w, w)
    if erase_h <= 0 or erase_w <= 0:
        return image
    top = int(torch.randint(0, max(1, h - erase_h + 1), (1,), device=image.device).item())
    left = int(torch.randint(0, max(1, w - erase_w + 1), (1,), device=image.device).item())
    out = image.clone()
    out[:, top : top + erase_h, left : left + erase_w] = erase_value
    return out


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    c, _, _ = image.shape
    if c == 1:
        return image
    gray = image.mean(dim=0, keepdim=True)
    return gray.repeat(c, 1, 1)


@dataclass
class LocalAugmentConfig:
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    color_jitter: Tuple[float, float] = (0.7, 1.3)
    erase_prob: float = 0.3
    erase_scale: Tuple[float, float] = (0.02, 0.15)
    gray_prob: float = 0.3


def _apply_random_local_augment(
    image: torch.Tensor,
    cfg: LocalAugmentConfig,
) -> torch.Tensor:
    """Apply one random augmentation from the configured set."""
    ops: List[Callable[[torch.Tensor], torch.Tensor]] = []
    sigma = float(_sample_uniform((1,), cfg.blur_sigma[0], cfg.blur_sigma[1], image.device, image.dtype))
    ops.append(lambda img: _gaussian_blur(img, sigma))

    cj_low, cj_high = cfg.color_jitter
    brightness = float(_sample_uniform((1,), cj_low, cj_high, image.device, image.dtype))
    contrast = float(_sample_uniform((1,), cj_low, cj_high, image.device, image.dtype))
    saturation = float(_sample_uniform((1,), cj_low, cj_high, image.device, image.dtype))
    ops.append(lambda img: _color_jitter(img, brightness, contrast, saturation))

    if torch.rand((), device=image.device) < cfg.erase_prob:
        ops.append(lambda img: _random_erasing(img, cfg.erase_scale))

    if torch.rand((), device=image.device) < cfg.gray_prob:
        ops.append(_to_grayscale)

    if not ops:
        return image
    op_idx = int(torch.randint(0, len(ops), (1,), device=image.device).item())
    return ops[op_idx](image)


def _boxes_to_masks(
    boxes: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(width, int(round(x2)))
        y2 = min(height, int(round(y2)))
        mask = torch.zeros((height, width), device=device, dtype=dtype)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
        masks.append(mask)
    if not masks:
        return torch.zeros((0, height, width), device=device, dtype=dtype)
    return torch.stack(masks, dim=0)


class GlobalLocalTransformation(nn.Module):

    def __init__(
        self,
        r_range: Tuple[float, float] = (0.05, 0.5),
        band_width: Optional[float] = 0.1,
        alpha_range: Tuple[float, float] = (0.0, 1.0),
        clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        local_cfg: Optional[LocalAugmentConfig] = None,
        use_sam: bool = False,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h"
    ) -> None:
        super().__init__()
        self.r_range = r_range
        self.band_width = band_width
        self.alpha_range = alpha_range
        self.clamp_range = clamp_range
        self.local_cfg = local_cfg or LocalAugmentConfig()

        self.use_sam = use_sam and SAM_AVAILABLE
        self.sam_mask_generator = None
        if self.use_sam:
            print(f"Initializing SAM for GLT...")
            self.sam_generator = SAMMaskGenerator(
                checkpoint_path=sam_checkpoint, 
                model_type=sam_model_type,
                device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            

    def global_transform(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        device = images.device
        dtype = images.dtype
        r = _sample_uniform((b,), self.r_range[0], self.r_range[1], device, dtype)
        out = []
        for i in range(b):
            mask = _build_band_pass_mask(h, w, float(r[i]), self.band_width, device, dtype)
            mask = mask[None, None, :, :]  # (1,1,H,W)
            fft = torch.fft.fft2(images[i : i + 1], dim=(-2, -1))
            noise = torch.randn_like(fft)
            randomized = fft * (1.0 + noise)
            blended = mask * randomized + (1.0 - mask) * fft
            inv = torch.fft.ifft2(blended, dim=(-2, -1)).real
            out.append(inv)
        out = torch.cat(out, dim=0)
        if self.clamp_range is not None:
            out = out.clamp(self.clamp_range[0], self.clamp_range[1])
        return out

    def local_transform(
        self,
        images: torch.Tensor,
        boxes: Optional[Sequence[torch.Tensor]] = None,
        masks: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, c, h, w = images.shape
        device = images.device
        dtype = images.dtype
        out = []
        for i in range(b):
            img = images[i]
            img_masks: Optional[torch.Tensor] = None
            
            # Case A: Masks already provided
            if masks is not None and i < len(masks):
                img_masks = masks[i].to(device=device, dtype=dtype)
                
            # Case B: Use SAM (External)
            elif self.use_sam and self.sam_generator is not None and boxes is not None and i < len(boxes):
                if len(boxes[i]) > 0:
                    # Convert Tensor (C, H, W) -> Numpy (H, W, C) uint8
                    img_np = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    
                    # CALL YOUR CUSTOM FUNCTION: generate_mask (singular)
                    sam_masks = self.sam_generator.generate_mask(img_np, boxes[i])
                    
                    # Convert back to tensor (float)
                    img_masks = sam_masks.to(device=device, dtype=dtype)
                else:
                     img_masks = torch.zeros((0, h, w), device=device, dtype=dtype)

            # Case C: Fallback to Box approximation
            elif boxes is not None and i < len(boxes):
                img_masks = _boxes_to_masks(boxes[i], h, w, device, dtype)

            # 2. Augmentation Logic (Applied to Objects vs Background)
            if img_masks is None or img_masks.numel() == 0:
                aug = _apply_random_local_augment(img, self.local_cfg)
                out.append(aug.unsqueeze(0))
                continue

            obj_masks = (img_masks > 0.5).to(dtype)
            union = obj_masks.sum(dim=0, keepdim=True).clamp(0, 1)
            bg_mask = (1.0 - union).expand(c, h, w)

            bg = img * bg_mask
            bg_aug = _apply_random_local_augment(bg, self.local_cfg) * bg_mask

            obj_accum = torch.zeros_like(img)
            # Sum augmentations for each object
            for mask in obj_masks:
                mask_c = mask.expand(c, h, w)
                obj = img * mask_c
                obj_aug = _apply_random_local_augment(obj, self.local_cfg) * mask_c
                obj_accum = obj_accum + obj_aug

            out.append((bg_aug + obj_accum).unsqueeze(0))

        out = torch.cat(out, dim=0)
        if self.clamp_range is not None:
            out = out.clamp(self.clamp_range[0], self.clamp_range[1])
        return out

    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[Sequence[torch.Tensor]] = None,
        masks: Optional[Sequence[torch.Tensor]] = None,
        return_parts: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt = self.global_transform(images)
        lt = self.local_transform(images, boxes=boxes, masks=masks)
        alpha = _sample_uniform((images.shape[0], 1, 1, 1), self.alpha_range[0], self.alpha_range[1], images.device, images.dtype)
        glt = alpha * gt + (1.0 - alpha) * lt
        if self.clamp_range is not None:
            glt = glt.clamp(self.clamp_range[0], self.clamp_range[1])
        if return_parts:
            return glt, gt, lt
        return glt


class CausalAttentionLearning(nn.Module):
    """Causal Attention Learning (CAL) module with Dice loss."""

    def __init__(
        self,
        binarize_threshold: Optional[float] = None,
        dice_eps: float = 1.0,
        use_ste: bool = True,
    ) -> None:
        super().__init__()
        self.binarize_threshold = binarize_threshold
        self.dice_eps = dice_eps
        self.use_ste = use_ste

    @staticmethod
    def attention_map(features: torch.Tensor) -> torch.Tensor:
        # Eq. (8): F_att = sigmoid(E(x)).
        return torch.sigmoid(features)

    def binarize(self, att: torch.Tensor) -> torch.Tensor:
        if self.binarize_threshold is None:
            thresh = att.mean(dim=(2, 3), keepdim=True)
        else:
            thresh = torch.tensor(self.binarize_threshold, device=att.device, dtype=att.dtype)
        hard = (att >= thresh).to(att.dtype)
        if self.use_ste:
            # Straight-through estimator: hard mask in forward, identity in backward.
            return hard + (att - att.detach())
        return hard

    def dice_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        intersection = (a * b).sum(dim=(1, 2, 3))
        union = a.sum(dim=(1, 2, 3)) + b.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.dice_eps) / (union + self.dice_eps)
        return 1.0 - dice.mean()

    def forward(self, features_orig: torch.Tensor, features_aug: torch.Tensor) -> torch.Tensor:
        att_orig = self.binarize(self.attention_map(features_orig))
        att_aug = self.binarize(self.attention_map(features_aug))
        return self.dice_loss(att_orig, att_aug)

    @staticmethod
    def apply_attention(features: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        return features * att_map


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def _compute_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute class prototypes by averaging ROI features per class.

    Background (label 0) is excluded from prototype computation.
    """
    device = features.device
    dtype = features.dtype
    feat_dim = features.shape[-1]
    valid = (labels > 0) & (labels < num_classes)
    if not valid.any().item():
        protos = torch.zeros((num_classes, feat_dim), device=device, dtype=dtype)
        present = torch.zeros((num_classes,), device=device, dtype=torch.bool)
        return protos, present

    features = features[valid]
    labels = labels[valid]
    protos = torch.zeros((num_classes, feat_dim), device=device, dtype=dtype)
    counts = torch.zeros((num_classes,), device=device, dtype=dtype)
    protos.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=dtype))
    counts_clamped = counts.clamp_min(1.0).unsqueeze(-1)
    protos = protos / counts_clamped
    present = counts > 0
    return protos, present


def _prototype_contrastive_loss(
    protos_src: torch.Tensor,
    present_src: torch.Tensor,
    protos_aug: torch.Tensor,
    present_aug: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    common = present_src & present_aug
    if not common.any():
        return torch.tensor(0.0, device=protos_src.device, dtype=protos_src.dtype)

    src_indices = torch.nonzero(common, as_tuple=False).squeeze(1)
    aug_indices = torch.nonzero(present_aug, as_tuple=False).squeeze(1)

    src = F.normalize(protos_src[src_indices], dim=-1)
    aug = F.normalize(protos_aug[aug_indices], dim=-1)

    logits = (src @ aug.t()) / temperature
    target_map = {int(cls_id): idx for idx, cls_id in enumerate(aug_indices.tolist())}
    targets = torch.tensor([target_map[int(cls_id)] for cls_id in src_indices.tolist()], device=logits.device)
    return F.cross_entropy(logits, targets)


class CausalPrototypeLearning(nn.Module):
    """Causal Prototype Learning (CPL) module."""

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.07,
        detach_target: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.detach_target = detach_target

    def explicit_loss(self, logits_src: torch.Tensor, logits_aug: torch.Tensor) -> torch.Tensor:
        p_src = F.softmax(logits_src, dim=-1)
        p_aug = F.softmax(logits_aug, dim=-1)
        if self.detach_target:
            p_src = p_src.detach()
        return _kl_divergence(p_src, p_aug)

    def implicit_loss(self, feats_src: torch.Tensor, feats_aug: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        protos_src, present_src = _compute_prototypes(feats_src, labels, self.num_classes)
        protos_aug, present_aug = _compute_prototypes(feats_aug, labels, self.num_classes)
        return _prototype_contrastive_loss(
            protos_src,
            present_src,
            protos_aug,
            present_aug,
            temperature=self.temperature,
        )

    def forward(
        self,
        logits_src: torch.Tensor,
        logits_aug: torch.Tensor,
        feats_src: torch.Tensor,
        feats_aug: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_exp = self.explicit_loss(logits_src, logits_aug)
        l_imp = self.implicit_loss(feats_src, feats_aug, labels)
        l_prot = l_exp + l_imp
        return l_prot, l_exp, l_imp


def sdg_total_loss(
    supervised_loss: torch.Tensor,
    attention_loss: torch.Tensor,
    prototype_loss: torch.Tensor,
    lambda_att: float = 0.1,
    lambda_prot: float = 0.1,
) -> torch.Tensor:
    """Total loss from Eq. (16)."""
    return supervised_loss + lambda_att * attention_loss + lambda_prot * prototype_loss


