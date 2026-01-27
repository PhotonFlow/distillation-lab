import math
from typing import Iterable

import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.util.misc import NestedTensor

try:
    from torch.amp import GradScaler, autocast
    DEPRECATED_AMP = False
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler, autocast
    DEPRECATED_AMP = True


RFDETR_MEAN = (0.485, 0.456, 0.406)
RFDETR_STD = (0.229, 0.224, 0.225)


def enable_pgd_training() -> None:
    import rfdetr.engine as engine
    import rfdetr.main as main

    if getattr(engine, "_pgd_patched", False):
        return

    engine._pgd_patched = True
    engine._orig_train_one_epoch = engine.train_one_epoch
    main._orig_train_one_epoch = getattr(main, "train_one_epoch", None)
    engine.train_one_epoch = train_one_epoch_pgd
    main.train_one_epoch = train_one_epoch_pgd
    print("[PGD] RF-DETR train_one_epoch patched.")


def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": torch.bfloat16}
    return {"device_type": "cuda", "enabled": args.amp, "dtype": torch.bfloat16}


def _pgd_bounds(device, dtype, eps, alpha):
    mean = torch.tensor(RFDETR_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(RFDETR_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    eps_norm = eps / std
    alpha_norm = alpha / std
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    return eps_norm, alpha_norm, min_val, max_val


def _pgd_attack(model, criterion, samples: NestedTensor, targets, args) -> NestedTensor:
    clean = samples.tensors.detach()
    device = clean.device
    dtype = clean.dtype

    eps = float(getattr(args, "adv_eps", 4.0 / 255.0))
    alpha = float(getattr(args, "adv_alpha", 2.0 / 255.0))
    steps = int(getattr(args, "adv_steps", 3))
    random_start = bool(getattr(args, "adv_random_start", True))

    eps_norm, alpha_norm, min_val, max_val = _pgd_bounds(device, dtype, eps, alpha)

    adv = clean.clone()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-1.0, 1.0) * eps_norm
    adv = torch.max(torch.min(adv, clean + eps_norm), clean - eps_norm)
    adv = torch.clamp(adv, min=min_val, max=max_val)

    for _ in range(steps):
        adv.requires_grad_(True)
        adv_samples = NestedTensor(adv, samples.mask)
        with autocast(**get_autocast_args(args)):
            outputs = model(adv_samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() + alpha_norm * grad.sign()
        adv = torch.max(torch.min(adv, clean + eps_norm), clean - eps_norm)
        adv = torch.clamp(adv, min=min_val, max=max_val)

    return NestedTensor(adv.detach(), samples.mask)


def train_one_epoch_pgd(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks=None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler("cuda", enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    adv_enabled = bool(getattr(args, "adv_enabled", False)) and epoch >= int(getattr(args, "adv_start_epoch", 0))
    adv_ratio = float(getattr(args, "adv_ratio", 0.5))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {"step": it, "model": model, "epoch": epoch}
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)

        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            from rfdetr.datasets.coco import compute_multi_scale_scales
            scales = compute_multi_scale_scales(
                args.resolution, args.expanded_scales, args.patch_size, args.num_windows
            )
            import random
            random.seed(it)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = F.interpolate(
                    samples.mask.unsqueeze(1).float(), size=scale, mode="nearest"
                ).squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                clean_loss = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )

            if adv_enabled:
                adv_samples = _pgd_attack(model, criterion, new_samples, new_targets, args)
                with autocast(**get_autocast_args(args)):
                    adv_outputs = model(adv_samples, new_targets)
                    adv_loss_dict = criterion(adv_outputs, new_targets)
                    adv_loss = sum(
                        (1 / args.grad_accum_steps) * adv_loss_dict[k] * weight_dict[k]
                        for k in adv_loss_dict.keys()
                        if k in weight_dict
                    )
                total_loss = (1 - adv_ratio) * clean_loss + adv_ratio * adv_loss
            else:
                total_loss = clean_loss

            scaler.scale(total_loss).backward()

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
