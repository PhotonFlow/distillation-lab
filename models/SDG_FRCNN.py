import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.feature_adapter import (GlobalLocalTransformation,
                            CausalAttentionLearning,
                            CausalPrototypeLearning,
                            sdg_total_loss
                            )


class SDG_FRCNN(nn.Module):
    def __init__(self, num_classes,use_sam=False, sam_checkpoint=None):
        super().__init__()
        self.base_model=fasterrcnn_resnet50_fpn(pretrained=True)
        in_features=self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor=FastRCNNPredictor(in_features, num_classes)

        old_heads=self.base_model.roi_heads
        self.base_model.roi_heads=RoIHeads(
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
            detections_per_img=old_heads.detections_per_img
        )

        self.glt=GlobalLocalTransformation(
            use_sam=use_sam
         )

        self.cal=CausalAttentionLearning()
        self.cpl=CausalPrototypeLearning(num_classes=num_classes)

        self.backbone_features=None
        self.base_model.backbone.register_forward_hook(self._hook_backbone)



    def _hook_backbone(self, module, input, output):
        # FPN output is dict {'0':, '1':, ...}. We use '0' (highest res) for Attention
        if isinstance(output, dict):
            self.backbone_features = list(output.values())[0]
        else:
            self.backbone_features = output

    
    def forward(self, images, targets=None):
        if self.training:
            # --- 1. Global-Local Transformation ---
            # Stack images for processing (assuming same size, handled by GLT internal)
            # But typically R-CNN takes list[Tensor]. 
            # We stack them temporarily for GLT if they are same size, 
            # or loop inside GLT. Your GLT expects tensor (B, C, H, W).
            # FasterRCNN images are usually different sizes, so we must be careful.
            # For simplicity here, we assume the collate_fn padded them or we resize.
            
            # Convert List[Tensor] -> Tensor (B, C, H, W)
            # NOTE: This requires images to be resized to same shape! 
            # If your dataset doesn't resize, this stack will fail.
            # Using NestedTensor or padding is safer, but for GLT we assume padded batch.
            from torch.nn.utils.rnn import pad_sequence
            # Simple padding strategy
            # ... (Implementation depends on your collate function)
            
            # Assuming images are already a tensor or we use the first one as ref
            # Let's try to stack. If fails, user must ensure resize in Transform.
            try:
                imgs_stack = torch.stack(images)
            except:
                raise ValueError("SDG: Images in batch must have same size for GLT augmentation.")

            # Get Augmented Views
            images_aug_tensor = self.glt(imgs_stack, boxes=[t['boxes'] for t in targets])
            images_aug_list = [img for img in images_aug_tensor]

            # --- 2. Pass 1: Source (Original) ---
            loss_dict_src = self.base_model(images, targets)
            
            # Capture features
            feat_src = self.backbone_features
            roi_feat_src = self.base_model.roi_heads.cur_box_features
            logits_src = self.base_model.roi_heads.cur_logits
            labels_src = torch.cat(self.base_model.roi_heads.cur_labels)
            
            # --- 3. Pass 2: Augmented ---
            # We fix targets (SDG assumes labels don't change with augmentation)
            loss_dict_aug = self.base_model(images_aug_list, targets)
            
            feat_aug = self.backbone_features
            roi_feat_aug = self.base_model.roi_heads.cur_box_features
            logits_aug = self.base_model.roi_heads.cur_logits
            labels_aug = torch.cat(self.base_model.roi_heads.cur_labels) # Likely different samples than src

            # --- 4. Calculate SDG Losses ---
            
            # A. Causal Attention Loss (Dice)
            # feat_src and feat_aug should be aligned if GLT is pixel-wise (no geometric shift)
            l_att = self.cal(feat_src, feat_aug)
            
            # B. Causal Prototype Loss
            # We use Implicit Loss (Contrastive on Prototypes)
            # We calculate prototypes for Source batch and Aug batch independently
            # and minimize distance between prototypes of same class.
            l_prot, l_exp, l_imp = self.cpl(
                logits_src, logits_aug,
                roi_feat_src, roi_feat_aug,
                labels_src # We use labels_src for src prototypes. 
                           # Ideally we pass labels_aug for aug prototypes inside CPL if they differ.
                           # Current CPL code assumes one label set? 
                           # Let's check CPL signature: implicit_loss(feats_src, feats_aug, labels)
                           # It assumes 'labels' applies to BOTH. 
                           # BUT Faster RCNN samples different ROIs for src and aug runs.
                           # FIX: We need to modify CPL call or pass to calculate prototypes separately.
            )
            
            # FIX for CPL with different proposals:
            # We manually call internal prototype computation since proposals differ
            from feature_adapter import _compute_prototypes, _prototype_contrastive_loss
            
            # Prototypes Source
            protos_src, present_src = _compute_prototypes(roi_feat_src, labels_src, self.cpl.num_classes)
            # Prototypes Aug
            protos_aug, present_aug = _compute_prototypes(roi_feat_aug, labels_aug, self.cpl.num_classes)
            
            l_imp_correct = _prototype_contrastive_loss(
                protos_src, present_src, protos_aug, present_aug, self.cpl.temperature
            )
            
            # Explicit Loss (KL) - requires matched logits.
            # Since proposals differ, we cannot do element-wise KL on logits.
            # We skip l_exp or set to 0 unless we force fixed proposals (very hard in R-CNN).
            l_exp_correct = torch.tensor(0.0).to(l_imp_correct.device) 
            
            l_prot_final = l_imp_correct
            
            # --- 5. Total Loss ---
            sup_loss = sum(loss for loss in loss_dict_src.values())
            total_loss = sdg_total_loss(sup_loss, l_att, l_prot_final)
            
            return {
                "total_loss": total_loss,
                "sup_loss": sup_loss,
                "att_loss": l_att,
                "prot_loss": l_prot_final
            }

        else:
            # Inference
            return self.base_model(images)


class SDGRoIHeads(RoIHeads):
    """
    Custom RoIHeads that exposes intermediate features and labels
    required for Causal Prototype Learning (CPL).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Storage for the current forward pass
        self.cur_box_features = None
        self.cur_labels = None
        self.cur_logits = None

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Modified forward pass to cache features and labels.
        Logic mostly copied from torchvision.models.detection.roi_heads.RoIHeads.forward
        """
        if self.training:
            # 1. Sample proposals (foreground/background)
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # 2. Extract features using RoI Align
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        
        # 3. Pass through Head (FC layers)
        box_features = self.box_head(box_features)
        
        # 4. Predict (Classification + BBox Reg)
        class_logits, box_regression = self.box_predictor(box_features)

        # --- CAPTURE FOR SDG ---
        if self.training:
            self.cur_box_features = box_features
            self.cur_labels = [l for l in labels] # List of tensors
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

# Helper to replicate private torchvision function
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    from torch.nn import functional as F
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
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





