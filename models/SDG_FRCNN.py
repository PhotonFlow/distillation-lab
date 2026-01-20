import torch
import torch.nn as nn
import types
from collections import OrderedDict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

# Import your modules
from models.feature_adapter import (
    GlobalLocalTransformation,
    CausalAttentionLearning,
    CausalPrototypeLearning,
    sdg_total_loss
)

class SDGRoIHeads(RoIHeads):
    """
    Custom RoIHeads that exposes intermediate features and labels
    required for Causal Prototype Learning (CPL).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize storage for the current forward pass
        self.cur_box_features = None
        self.cur_labels = None
        self.cur_logits = None
        self.cur_rpn_scores = None
        self.cur_sampled_scores = None

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        rpn_counts = [p.shape[0] for p in proposals]

        # append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        sampled_scores = [] if self.cur_rpn_scores is not None else None
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            if sampled_scores is not None:
                scores_img = self.cur_rpn_scores[img_id]
                num_rpn = rpn_counts[img_id]
                if scores_img is not None and scores_img.numel() > num_rpn:
                    scores_img = scores_img[:num_rpn]
                scores = torch.zeros(
                    (img_sampled_inds.shape[0],),
                    device=device,
                    dtype=scores_img.dtype if scores_img is not None else torch.float32,
                )
                if scores_img is not None and scores_img.numel() > 0:
                    rpn_mask = img_sampled_inds < num_rpn
                    if rpn_mask.any():
                        scores[rpn_mask] = scores_img[img_sampled_inds[rpn_mask]]
                sampled_scores.append(scores)

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        self.cur_sampled_scores = sampled_scores
        return proposals, matched_idxs, labels, regression_targets

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Modified forward pass to cache features and labels.
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        # --- CAPTURE FOR SDG ---
        if self.training:
            self.cur_box_features = box_features
            self.cur_labels = labels 
            self.cur_logits = class_logits
        # -----------------------

        result = torch.jit.annotate(list[dict[str, torch.Tensor]], [])
        losses = {}

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses


class AttentionGuidedRPN(nn.Module):
    """Wrap an RPN to apply attention masks to backbone features before proposals."""

    def __init__(
        self,
        rpn: nn.Module,
        attention: "CausalAttentionLearning",
        binarize: bool = True,
        score_target=None,
    ) -> None:
        super().__init__()
        self.rpn = rpn
        self.attention = attention
        self.binarize = binarize
        self.score_target = score_target
        self.last_scores = None
        self._wrap_filter_proposals()

    def _wrap_filter_proposals(self):
        if not hasattr(self.rpn, "filter_proposals"):
            return
        orig_filter = self.rpn.filter_proposals
        outer = self

        def _filter(self_rpn, proposals, objectness, image_shapes, num_anchors_per_level):
            boxes, scores = orig_filter(proposals, objectness, image_shapes, num_anchors_per_level)
            outer.last_scores = scores
            if outer.score_target is not None:
                outer.score_target.cur_rpn_scores = scores
            return boxes, scores

        self.rpn.filter_proposals = types.MethodType(_filter, self.rpn)

    def _apply_attention(self, features):
        if isinstance(features, dict):
            out = features.__class__()
            for key, feat in features.items():
                att = self.attention.attention_map(feat)
                if self.binarize:
                    att = self.attention.binarize(att)
                out[key] = CausalAttentionLearning.apply_attention(feat, att)
            return out

        guided = []
        for feat in features:
            att = self.attention.attention_map(feat)
            if self.binarize:
                att = self.attention.binarize(att)
            guided.append(CausalAttentionLearning.apply_attention(feat, att))
        return guided

    def forward(self, images, features, targets=None):
        guided_features = self._apply_attention(features)
        return self.rpn(images, guided_features, targets)

    def __getattr__(self, name: str):
        if name in {"rpn", "attention", "binarize", "score_target", "last_scores"}:
            return super().__getattr__(name)
        return getattr(self.rpn, name)

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    from torch.nn import functional as F
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.shape[1] // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class SDGFasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        use_sam=False,
        sam_checkpoint=None,
        use_attention_rpn: bool = True,
        rpn_binarize: bool = True,
        explicit_rpn_thresh: float = 0.7,
    ):
        super().__init__()
        
        # 1. Initialize Standard Faster R-CNN
        print("Initializing Base Faster R-CNN...")
        self.base_model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # 2. Replace the Predictor (Head)
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 3. Inject Custom RoIHeads
        old_heads = self.base_model.roi_heads
        
        print("Swapping standard RoIHeads with SDGRoIHeads...")
        self.base_model.roi_heads = SDGRoIHeads(
            box_roi_pool=old_heads.box_roi_pool,
            box_head=old_heads.box_head,
            box_predictor=old_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=old_heads.score_thresh,
            nms_thresh=old_heads.nms_thresh,
            detections_per_img=old_heads.detections_per_img,
        )
        
        # Verify Swap
        if not isinstance(self.base_model.roi_heads, SDGRoIHeads):
            raise RuntimeError("CRITICAL ERROR: RoIHeads swap failed! The model is still using the standard class.")
        print("RoIHeads Swap Successful.")

        # 4. Initialize SDG Modules
        self.glt = GlobalLocalTransformation(
            use_sam=use_sam,
            sam_checkpoint=sam_checkpoint
        )
        self.cal = CausalAttentionLearning()
        self.cpl = CausalPrototypeLearning(num_classes=num_classes)

        # 5. Attention-guided RPN (Eq. 10)
        self.use_attention_rpn = use_attention_rpn
        self.rpn_binarize = rpn_binarize
        if self.use_attention_rpn:
            self.base_model.rpn = AttentionGuidedRPN(
                self.base_model.rpn,
                self.cal,
                binarize=self.rpn_binarize,
                score_target=self.base_model.roi_heads,
            )
        self.explicit_rpn_thresh = explicit_rpn_thresh
    
    def _forward_base(self, images, targets):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            if len(val) != 2:
                raise ValueError("Expected images to be a list of tensors with shape (C, H, W).")
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.base_model.transform(images, targets)
        features = self.base_model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.base_model.rpn(images, features, targets)
        rpn_scores = None
        if isinstance(self.base_model.rpn, AttentionGuidedRPN):
            rpn_scores = self.base_model.rpn.last_scores

        detections, detector_losses = self.base_model.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return {
            "images": images,
            "features": features,
            "proposals": proposals,
            "rpn_scores": rpn_scores,
            "losses": losses,
        }

    @staticmethod
    def _filter_proposals_by_score(proposals, scores, thresh):
        if scores is None:
            return proposals
        filtered = []
        for props, sc in zip(proposals, scores):
            if sc.numel() == 0:
                filtered.append(props[:0])
                continue
            keep = sc >= thresh
            filtered.append(props[keep])
        return filtered

    def _roi_logits(self, features, proposals, image_shapes):
        if sum(p.shape[0] for p in proposals) == 0:
            return None
        box_features = self.base_model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.base_model.roi_heads.box_head(box_features)
        class_logits, _ = self.base_model.roi_heads.box_predictor(box_features)
        return class_logits


    def forward(self, images, targets=None):
        if self.training:
            # 1. Check Image Sizes for Stack Error
            try:
                imgs_stack = torch.stack(images)
            except RuntimeError:
                raise ValueError("SDG Error: Batch images have different sizes. You MUST resize images to a fixed size (e.g. 640x640) in your Dataset class.")

            t_boxes = [t['boxes'] for t in targets] if targets else None
            images_aug_tensor = self.glt(imgs_stack, boxes=t_boxes)
            # if np.random.rand() < 0.01: # Save 1% of images to avoid spamming
            #     import torchvision.utils as vutils
            #     debug_grid = vutils.make_grid(
            #         torch.cat([imgs_stack[0:1], images_aug_tensor[0:1]], dim=0), 
            #         nrow=2, padding=5, normalize=False
            #     )
            #     vutils.save_image(debug_grid, f"debug_aug_{np.random.randint(1000)}.jpg")
            #     print("Saved debug_aug.jpg")
            images_aug_list = [img for img in images_aug_tensor]

            # 2. Pass 1: Source
            src_pass = self._forward_base(images, targets)
            feat_src = list(src_pass["features"].values())[0]
            
            # --- DEBUG CHECK ---
            if not hasattr(self.base_model.roi_heads, 'cur_box_features'):
                raise AttributeError(f"Model head is {type(self.base_model.roi_heads)}. It should be SDGRoIHeads.")
            # -------------------

            roi_feat_src = self.base_model.roi_heads.cur_box_features
            logits_src = self.base_model.roi_heads.cur_logits
            labels_src = torch.cat(self.base_model.roi_heads.cur_labels)
            proposals_src = src_pass["proposals"]
            scores_src_list = src_pass["rpn_scores"]
            
            # 3. Pass 2: Augmented
            aug_pass = self._forward_base(images_aug_list, targets)
            feat_aug = list(aug_pass["features"].values())[0]
            roi_feat_aug = self.base_model.roi_heads.cur_box_features
            logits_aug = self.base_model.roi_heads.cur_logits
            labels_aug = torch.cat(self.base_model.roi_heads.cur_labels) 

            # 4. Losses
            l_att = self.cal(feat_src, feat_aug)

            # Prototype loss = explicit (KL) + implicit (contrastive prototypes)
            proposals_exp = self._filter_proposals_by_score(
                proposals_src, scores_src_list, self.explicit_rpn_thresh
            )
            logits_src_exp = self._roi_logits(
                src_pass["features"], proposals_exp, src_pass["images"].image_sizes
            )
            logits_aug_exp = self._roi_logits(
                aug_pass["features"], proposals_exp, src_pass["images"].image_sizes
            )
            if logits_src_exp is None or logits_aug_exp is None:
                l_prot_exp = torch.tensor(0.0, device=logits_src.device, dtype=logits_src.dtype)
            else:
                l_prot_exp = self.cpl.explicit_loss(logits_src_exp, logits_aug_exp)

            from models.feature_adapter import _compute_prototypes, _prototype_contrastive_loss
            protos_src, present_src = _compute_prototypes(roi_feat_src, labels_src, self.cpl.num_classes)
            protos_aug, present_aug = _compute_prototypes(roi_feat_aug, labels_aug, self.cpl.num_classes)

            l_prot_imp = _prototype_contrastive_loss(
                protos_src, present_src, protos_aug, present_aug, self.cpl.temperature
            )
            l_prot_final = l_prot_exp + l_prot_imp
            
            sup_loss = sum(loss for loss in src_pass["losses"].values())
            total_loss = sdg_total_loss(sup_loss, l_att, l_prot_final)
            
            return {
                "total_loss": total_loss,
                "sup_loss": sup_loss,
                "att_loss": l_att,
                "prot_loss": l_prot_final,
                "prot_exp": l_prot_exp,
                "prot_imp": l_prot_imp
            }

        else:
            return self.base_model(images)




