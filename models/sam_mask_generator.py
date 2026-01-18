import torch 
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor



class SAMMaskGenerator:
    def __init__(self, checkpoint_path="sam_vit_h_4b8939.pth",model_type="vit_h", device="cuda"):
        self.device = device
        sam=sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)



    def generate_mask(self,image, boxes):
        self.predictor.set_image(image) # [H,W,3]
        if isinstance(boxes, torch.Tensor):  # [N, 4]
            boxes=boxes.cpu().numpy()
        transformed_boxes=self.predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.device), image.shape[:2]
            )
        masks,_,_=self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.squeeze(1)  # [N, H, W]










