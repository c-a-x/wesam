import os
import time
import torch
import lightning as L
import torch.nn.functional as F
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, calc_iou, validate
from utils.tools import copy_model, create_csv, momentum_update, reduce_instances
from ifp_alignment import build_prompt_generator
from utils.prompt_policy import apply_prompt_policy


def _gt_positive_prompt(gt_mask: torch.Tensor, device: torch.device) -> dict | None:
    """Build one stable positive point from the interior of a GT mask."""
    mask = gt_mask.to(device=device, dtype=torch.float32)
    if mask.numel() == 0:
        return None
    while mask.ndim > 2:
        mask = mask.amax(dim=0)
    if mask.ndim != 2:
        return None
    mask = mask > 0
    if not bool(mask.any()):
        return None

    # Prefer pixels surviving a small erosion so the prompt is away from the
    # object boundary. Fall back to all foreground pixels for small objects.
    mask_4d = mask[None, None].float()
    eroded = -F.max_pool2d(-mask_4d, kernel_size=7, stride=1, padding=3)[0, 0] > 0.5
    candidates = torch.nonzero(mask & eroded, as_tuple=False)
    if candidates.numel() == 0:
        candidates = torch.nonzero(mask, as_tuple=False)

    center = candidates.float().mean(dim=0, keepdim=True)
    point_yx = candidates[((candidates.float() - center) ** 2).sum(dim=1).argmin()]
    point_xy = point_yx[[1, 0]].float().reshape(1, 1, 2)
    labels = torch.ones((1, 1), dtype=torch.int, device=device)
    return {"in_points": (point_xy, labels), "in_box": None}


def _replace_labeled_prompts(
    prompts: list[dict], gt_masks, has_gts: torch.Tensor, probability: float = 1.0,
) -> list[dict]:
    """Optionally replace labeled IFP points with GT interior points."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("labeled_gt_prompt_probability must be in [0, 1]")
    updated = list(prompts)
    for index, (gt_mask, has_gt) in enumerate(zip(gt_masks, has_gts)):
        if not bool(has_gt):
            continue
        if probability < 1.0 and bool(torch.rand((), device=gt_mask.device) >= probability):
            continue
        gt_prompt = _gt_positive_prompt(gt_mask, prompts[index]["in_points"][0].device)
        if gt_prompt is not None:
            updated[index] = gt_prompt
    return updated


def save_evaluated_checkpoint(
    fabric: L.Fabric,
    path: str,
    model: Model,
    *,
    source_model: str,
    epoch: int,
    mean_iou: float,
    mean_f1: float,
    optimizer: _FabricOptimizer | None = None,
) -> None:
    """Save an evaluated model together with enough metadata to select it later."""
    state = {
        "model": model,
        "source_model": source_model,
        "epoch": epoch,
        "mean_iou": float(mean_iou),
        "mean_f1": float(mean_f1),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer
    fabric.save(path, state)


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    dino,
    teacher_model: Model,
    anchor_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    image_prompt_path, mask_prompt_path, prompt_generator=None,
    eval_prompt_generator=None,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    contra_loss = ContraLoss()
    best_student_iou = float("-inf")

    for epoch in range(1, cfg.num_epochs + 1):
        # The Student must be in training mode after its validation pass.
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        teacher_losses = AverageMeter()
        anchor_losses = AverageMeter()
        contra_losses = AverageMeter()
        sup_losses = AverageMeter()
        ifp_consistency_losses = AverageMeter()
        ifp_consistency_counts = AverageMeter()
        labeled_counts = AverageMeter()
        unlabeled_counts = AverageMeter()
        pseudo_counts = AverageMeter()
        gated_counts = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, has_gts = data
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            hybrid_enabled = float(
                getattr(cfg, "ifp_consistency_weight", 0.0) or 0.0
            ) > 0.0
            if hybrid_enabled:
                # Main teacher/student path deliberately receives no prompt.
                # IFP is an auxiliary branch only, so prompt errors cannot
                # overwrite the no-prompt pseudo-label target.
                prompts = None
                ifp_prompts = (prompt_generator(images_weak, gt_masks)
                               if getattr(prompt_generator, "requires_gt_masks", False)
                               else prompt_generator(images_weak))
                ifp_prompts = apply_prompt_policy(
                    ifp_prompts, cfg.prompt_generator,
                    images_weak.shape[-2:], training=True
                )
            else:
                prompts = (prompt_generator(images_weak, gt_masks)
                           if getattr(prompt_generator, "requires_gt_masks", False)
                           else prompt_generator(images_weak))
                if (
                    cfg.prompt_generator.backend != "no-prompt"
                    and cfg.prompt_generator.labeled_prompt_mode in {"gt", "mixed"}
                ):
                    prompts = _replace_labeled_prompts(
                        prompts, gt_masks, has_gts,
                        cfg.prompt_generator.labeled_gt_prompt_probability,
                    )
                prompts = apply_prompt_policy(
                    prompts, cfg.prompt_generator, images_weak.shape[-2:], training=True
                )

            has_unlabeled = any(not bool(has_gt) for has_gt in has_gts)
            # The EMA Teacher is not passed through ``fabric.setup()``, so it
            # needs Fabric's autocast context explicitly. Student forward is
            # autocast by the Fabric-wrapped module itself.
            with torch.inference_mode(), fabric.autocast():
                if has_unlabeled:
                    if teacher_model is None:
                        raise RuntimeError("Unlabeled samples require an EMA Teacher model.")
                    if anchor_model is not None:
                        anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(images_weak, prompts)
                    else:
                        anchor_image_embeds = None
                        anchor_masks = [None] * batch_size
                        anchor_res_masks = [None] * batch_size
                    if (
                        prompt_generator is not None
                        and cfg.prompt_generator.iterative_pseudo.enabled
                    ):
                        iterative_cfg = cfg.prompt_generator.iterative_pseudo
                        (
                            _soft_image_embeds,
                            soft_masks,
                            _soft_iou_predictions,
                            _soft_res_masks,
                        ) = prompt_generator.iterative_teacher_pseudo_labels(
                            teacher_model,
                            images_weak,
                            max_iters=iterative_cfg.max_iters,
                            teacher_iou_threshold=iterative_cfg.teacher_iou_threshold,
                            min_new_pixels=iterative_cfg.min_new_pixels,
                            min_new_area_ratio=iterative_cfg.min_new_area_ratio,
                            max_mask_area_ratio=iterative_cfg.max_mask_area_ratio,
                            min_overlap_ratio=iterative_cfg.min_overlap_ratio,
                            max_new_area_ratio=iterative_cfg.max_new_area_ratio,
                        )
                    else:
                        _, soft_masks, _, _ = teacher_model(images_weak, prompts)
                    # The teacher cache is not needed after producing masks;
                    # retaining it increases the peak when the student runs.
                    teacher_model.image_embeddings = None
                    teacher_model.vision_pos_enc = None
                    teacher_model.high_res_feats = None
                else:
                    anchor_image_embeds = None
                    anchor_masks = [None] * batch_size
                    anchor_res_masks = [None] * batch_size
                    soft_masks = [None] * batch_size

            if hybrid_enabled:
                # Main no-prompt path is computed exactly as in the
                # no-prompt baseline. The auxiliary IFP path runs under
                # no_grad() and no longer updates the shared encoder/LoRA.
                # The auxiliary loss therefore cannot change the main
                # solution; it is used only for soft diagnostic gating.
                pred_image_embeds = model.encode(images_strong)
                pred_masks, iou_predictions, pred_res_masks = model.decode(
                    images_strong.shape[-2:], None, prompt_mode="none"
                )
                with torch.no_grad():
                    ifp_masks, _, _ = model.decode(
                        images_strong.shape[-2:], ifp_prompts, prompt_mode="point"
                    )
            else:
                pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_strong, prompts)   # student
            if cfg.contrast_weight <= 0 and not hybrid_enabled:
                # Avoid keeping a large feature tensor alive through the loss
                # loop when contrastive alignment is disabled.
                pred_image_embeds = None

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            loss_anchor = torch.tensor(0., device=fabric.device)
            loss_contra = torch.tensor(0., device=fabric.device)
            unlabeled_count = 0
            pseudo_count = 0
            gated_count = 0
            unlabeled_indices = []
            agreement_gate = float(
                getattr(
                    cfg.prompt_generator.iterative_pseudo,
                    "min_student_teacher_iou",
                    0.0,
                )
                or 0.0
            )

            for i, (pred_mask, soft_mask, anchor_mask, iou_prediction) in enumerate(zip(pred_masks, soft_masks, anchor_masks, iou_predictions)):
                if bool(has_gts[i]):
                    continue

                unlabeled_count += 1
                unlabeled_indices.append(i)

                if cfg.anchor_weight > 0:
                    if anchor_mask is None:
                        raise RuntimeError("Anchor loss requires an Anchor model.")
                    loss_anchor += dice_loss(pred_mask, (anchor_mask > 0.).float())

                soft_mask = (soft_mask > 0.).float()
                if soft_mask.sum().item() == 0:
                    continue

                # Reject pseudo supervision when the student and teacher
                # disagree: a low-agreement sample is exactly the case where
                # the teacher prompt/mask is untrustworthy, and fitting it
                # would reinforce the teacher's error.
                if agreement_gate > 0.0:
                    agreement = float(calc_iou(pred_mask, soft_mask).reshape(-1)[0].item())
                    if agreement < agreement_gate:
                        gated_count += 1
                        continue

                pseudo_count += 1
                loss_focal += focal_loss(pred_mask, soft_mask)
                loss_dice += dice_loss(pred_mask, soft_mask)
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum')

            # ContraLoss needs cross-image negatives. Calling it once per
            # image always gives a zero loss because its similarity matrix is
            # 1x1. Restrict it to the unlabeled subset to keep anchor/student
            # alignment independent of the GT-supervised samples.
            if cfg.contrast_weight > 0 and len(unlabeled_indices) >= 2:
                if anchor_image_embeds is None:
                    raise RuntimeError("Contrast loss requires an Anchor model.")
                indices = torch.tensor(unlabeled_indices, device=pred_image_embeds.device)
                loss_contra = contra_loss(
                    pred_image_embeds.index_select(0, indices),
                    anchor_image_embeds.index_select(0, indices),
                    torch.cat([pred_res_masks[i] for i in unlabeled_indices], dim=0),
                    torch.cat([anchor_res_masks[i] for i in unlabeled_indices], dim=0).detach(),
                )

            if pseudo_count > 0:
                loss_focal = loss_focal / pseudo_count
                loss_dice = loss_dice / pseudo_count
                loss_iou = loss_iou / pseudo_count
            if unlabeled_count > 0:
                loss_anchor = loss_anchor / unlabeled_count

            # GT 监督 loss: only exposed labeled samples use ground-truth masks.
            loss_sup = torch.tensor(0., device=fabric.device)
            labeled_count = 0
            for i, (pred_mask_i, gt_mask_item, has_gt) in enumerate(zip(pred_masks, gt_masks, has_gts)):
                if not bool(has_gt):
                    continue
                if gt_mask_item.dim() == 3 and gt_mask_item.shape[0] > 1:
                    gt_combined = (gt_mask_item.sum(dim=0) > 0).float()
                else:
                    gt_combined = gt_mask_item.squeeze(0).float()
                loss_sup += focal_loss(pred_mask_i, gt_combined) + dice_loss(pred_mask_i, gt_combined)
                labeled_count += 1

            if labeled_count > 0:
                loss_sup = loss_sup / labeled_count

            # Auxiliary consistency branch: IFP is fully detached from the
            # optimiser in v3. We still report its disagreement with the main
            # no-prompt target as a diagnostic, but add NO gradient term.
            loss_ifp_consistency = torch.tensor(0., device=fabric.device)
            ifp_consistency_count = 0
            if hybrid_enabled:
                with torch.no_grad():
                    for i, ifp_mask_i in enumerate(ifp_masks):
                        if bool(has_gts[i]):
                            gt_mask_item = gt_masks[i]
                            if gt_mask_item.dim() == 3 and gt_mask_item.shape[0] > 1:
                                target_i = (gt_mask_item.sum(dim=0) > 0).float()
                            else:
                                target_i = gt_mask_item.squeeze(0).float()
                        else:
                            if soft_masks[i] is None:
                                continue
                            target_i = (soft_masks[i] > 0).float()
                        if target_i.sum().item() == 0:
                            continue
                        loss_ifp_consistency += (
                            focal_loss(ifp_mask_i, target_i) +
                            dice_loss(ifp_mask_i, target_i)
                        )
                        ifp_consistency_count += 1
                    if ifp_consistency_count > 0:
                        loss_ifp_consistency = loss_ifp_consistency / ifp_consistency_count

            # The former 20x focal term made a single bad pseudo-mask produce
            # a much larger update than several correctly supervised samples.
            teacher_loss = loss_focal + loss_dice + 0.1 * loss_iou
            rampup_epochs = max(int(cfg.unsupervised_rampup_epochs), 1)
            unsup_ramp = min(1.0, epoch / rampup_epochs)
            loss_total = (unsup_ramp * cfg.teacher_weight * teacher_loss +
                          unsup_ramp * cfg.anchor_weight * loss_anchor +
                          unsup_ramp * cfg.contrast_weight * loss_contra +
                          cfg.supervised_weight * loss_sup)
            # v3 auxiliary branch is diagnostic-only and contributes no
            # gradient to the main no-prompt objective.
            # A batch can contain only unlabeled images whose Teacher masks
            # were all rejected. It has no supervised or pseudo-label signal.
            if loss_total.requires_grad:
                fabric.backward(loss_total)
                optimizer.step()
                if teacher_model is not None:
                    momentum_update(model, teacher_model, momentum=cfg.ema_rate)
                scheduler.step()
            optimizer.zero_grad()
            model.image_embeddings = None
            model.vision_pos_enc = None
            model.high_res_feats = None

            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            teacher_losses.update(teacher_loss.item(), batch_size)
            anchor_losses.update(loss_anchor.item(), batch_size)
            contra_losses.update(loss_contra.item(), batch_size)
            sup_losses.update(loss_sup.item(), batch_size)
            ifp_consistency_losses.update(loss_ifp_consistency.item(), batch_size)
            ifp_consistency_counts.update(ifp_consistency_count, batch_size)
            labeled_counts.update(labeled_count, batch_size)
            unlabeled_counts.update(unlabeled_count, batch_size)
            pseudo_counts.update(pseudo_count, batch_size)
            gated_counts.update(gated_count, batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Teacher Loss [{teacher_losses.val:.4f} ({teacher_losses.avg:.4f})]'
                         f' | Anchor Loss [{anchor_losses.val:.4f} ({anchor_losses.avg:.4f})]'
                         f' | Contrast Loss [{contra_losses.val:.4f} ({contra_losses.avg:.4f})]'
                         f' | Supervised Loss [{sup_losses.val:.4f} ({sup_losses.avg:.4f})]'
                         f' | IFP Consistency [{ifp_consistency_losses.val:.4f} ({ifp_consistency_losses.avg:.4f})]'
                         f' | IFP Count [{ifp_consistency_counts.val:.0f} ({ifp_consistency_counts.avg:.2f})]'
                         f' | Labeled Count [{labeled_counts.val:.0f} ({labeled_counts.avg:.2f})]'
                         f' | Unlabeled Count [{unlabeled_counts.val:.0f} ({unlabeled_counts.avg:.2f})]'
                         f' | Valid Pseudo [{pseudo_counts.val:.0f} ({pseudo_counts.avg:.2f})]'
                         f' | Gated Pseudo [{gated_counts.val:.0f} ({gated_counts.avg:.2f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Focal Loss": focal_losses.avg, "Dice Loss": dice_losses.avg,
                "IoU Loss": iou_losses.avg, "Teacher Loss": teacher_losses.avg,
                "Anchor Loss": anchor_losses.avg, "Contrast Loss": contra_losses.avg,
                "Supervised Loss": sup_losses.avg, "IFP Consistency": ifp_consistency_losses.avg,
                "IFP Count": ifp_consistency_counts.avg, "Labeled Count": labeled_counts.avg,
                "Unlabeled Count": unlabeled_counts.avg, "Valid Pseudo": pseudo_counts.avg,
                "Gated Pseudo": gated_counts.avg,
                "Total Loss": total_losses.avg}
            fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)

        if epoch % cfg.eval_interval == 0:
            student_iou, student_f1 = validate(
                fabric, cfg, model, dino, val_dataloader, image_prompt_path, mask_prompt_path,
                f"{cfg.name}_student", epoch,
                prompt_generator=eval_prompt_generator,
            )
            save_dir = os.path.join(cfg.out_dir, "save")
            if student_iou > best_student_iou:
                save_evaluated_checkpoint(
                    fabric, os.path.join(save_dir, "best-student.pth"), model,
                    source_model="student", epoch=epoch, mean_iou=student_iou,
                    mean_f1=student_f1, optimizer=optimizer,
                )
                best_student_iou = student_iou



def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        main(cfg)


def main(cfg: Box) -> None:
    # IFP supplies point prompts at runtime; legacy file-based prompt paths
    # are optional and remain unused by this retained workflow.
    image_prompt_path = getattr(cfg, "image_prompt_path", None)
    mask_prompt_path = getattr(cfg, "mask_prompt_path", None)
    cfg.out_dir = os.path.join(cfg.out_dir, cfg.dataset)

    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      precision=getattr(cfg, "precision", "32-true"),
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(int(cfg.semi.seed) + fabric.global_rank)

    prompt_generator = None
    if cfg.prompt_generator.backend in {"ifp", "dino-prototype", "clip-only", "no-prompt", "gt-oracle"}:
        prompt_generator = build_prompt_generator(
            cfg.prompt_generator.backend,
            fabric.device,
            checkpoint_path=cfg.prompt_generator.alignment_checkpoint,
            background_checkpoint_path=cfg.prompt_generator.background_alignment_checkpoint or None,
            text_prompts=cfg.prompt_generator.text_prompts,
            background_text_prompts=cfg.prompt_generator.background_text_prompts or None,
            dino_variant=cfg.prompt_generator.dino_variant,
            clip_variant=cfg.prompt_generator.clip_variant,
            ifp_root=cfg.prompt_generator.ifp_root,
            resize=cfg.prompt_generator.resize or None,
            score_threshold=cfg.prompt_generator.score_threshold,
            max_positive_points=cfg.prompt_generator.max_positive_points,
            min_positive_points=cfg.prompt_generator.min_positive_points,
            include_box=cfg.prompt_generator.include_box,
        )
        dino = None
    else:
        raise ValueError(f"Unsupported prompt backend: {cfg.prompt_generator.backend}")

    hybrid_enabled = float(getattr(cfg, "ifp_consistency_weight", 0.0) or 0.0) > 0.0
    eval_prompt_generator = None if hybrid_enabled else prompt_generator

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, model.image_size)
    optimizer, scheduler = configure_opt(cfg, model)

    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])

    fully_supervised = cfg.semi.labeled_ratio >= 1.0
    # The Anchor has no role when neither of its losses is enabled. Avoid
    # keeping a full frozen SAM2 copy and running its inference in that case.
    needs_anchor = cfg.anchor_weight > 0 or cfg.contrast_weight > 0
    anchor_model = None if fully_supervised or not needs_anchor else copy_model(model)
    teacher_model = None if fully_supervised else copy_model(model)

    validate(
        fabric, cfg, model, dino, val_data, image_prompt_path, mask_prompt_path, name=cfg.name, epoch=0,
        prompt_generator=eval_prompt_generator,
    )
    train_sam(
        cfg, fabric, model, dino, teacher_model, anchor_model, optimizer, scheduler, train_data, val_data,
        image_prompt_path, mask_prompt_path, prompt_generator=prompt_generator,
        eval_prompt_generator=eval_prompt_generator,
    )

    final_test_list = str(getattr(cfg.datasets[cfg.dataset], "final_test_list", ""))
    if final_test_list:
        best_path = os.path.join(cfg.out_dir, "save", "best-student.pth")
        checkpoint = fabric.load(best_path)
        model.load_state_dict(checkpoint["model"])
        cfg.datasets[cfg.dataset].test_list = final_test_list
        _, test_data = load_datasets(cfg, model.image_size)
        test_data = fabric._setup_dataloader(test_data)
        validate(
            fabric, cfg, model, dino, test_data, image_prompt_path, mask_prompt_path,
            name=f"{cfg.name}_final_test_best_student",
            epoch=int(checkpoint.get("epoch", cfg.num_epochs)),
            prompt_generator=eval_prompt_generator,
        )

    del model, teacher_model, anchor_model, train_data, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    main(cfg)
    torch.cuda.empty_cache()
