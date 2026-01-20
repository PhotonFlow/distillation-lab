import torch
import torch.nn as nn
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
    def __init__(self, num_classes, use_sam=False, sam_checkpoint=None):
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
        
        self.backbone_features = None
        self.base_model.backbone.register_forward_hook(self._hook_backbone)

    def _hook_backbone(self, module, input, output):
        if isinstance(output, dict):
            self.backbone_features = list(output.values())[0]
        else:
            self.backbone_features = output

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
            loss_dict_src = self.base_model(images, targets)
            
            feat_src = self.backbone_features
            
            # --- DEBUG CHECK ---
            if not hasattr(self.base_model.roi_heads, 'cur_box_features'):
                raise AttributeError(f"Model head is {type(self.base_model.roi_heads)}. It should be SDGRoIHeads.")
            # -------------------

            roi_feat_src = self.base_model.roi_heads.cur_box_features
            logits_src = self.base_model.roi_heads.cur_logits
            labels_src = torch.cat(self.base_model.roi_heads.cur_labels)
            
            # 3. Pass 2: Augmented
            loss_dict_aug = self.base_model(images_aug_list, targets)
            
            feat_aug = self.backbone_features
            roi_feat_aug = self.base_model.roi_heads.cur_box_features
            logits_aug = self.base_model.roi_heads.cur_logits
            labels_aug = torch.cat(self.base_model.roi_heads.cur_labels) 

            # 4. Losses
            l_att = self.cal(feat_src, feat_aug)
            
            from models.feature_adapter import _compute_prototypes, _prototype_contrastive_loss
            protos_src, present_src = _compute_prototypes(roi_feat_src, labels_src, self.cpl.num_classes)
            protos_aug, present_aug = _compute_prototypes(roi_feat_aug, labels_aug, self.cpl.num_classes)
            
            l_prot_final = _prototype_contrastive_loss(
                protos_src, present_src, protos_aug, present_aug, self.cpl.temperature
            )
            
            sup_loss = sum(loss for loss in loss_dict_src.values())
            total_loss = sdg_total_loss(sup_loss, l_att, l_prot_final)
            
            return {
                "total_loss": total_loss,
                "sup_loss": sup_loss,
                "att_loss": l_att,
                "prot_loss": l_prot_final
            }

        else:
            return self.base_model(images)




