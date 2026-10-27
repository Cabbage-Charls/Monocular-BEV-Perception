import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation)


# ADE20K 标准类别 ID（nvidia/segformer-b0-finetuned-ade-512-512）
SKY_CLASS = 2

# 近景类别：道路、人行道、行人、车、公交、卡车、摩托、自行车
# 这是初始值，跑 verify_seg.py 后按实际调整
KEEP_CLASSES_ADE = {3, 6, 9, 11, 12, 18, 19, 20, 32, 33}
SKY_CLASS = 2


class SemanticSegmenter:
    def __init__(self, device="cuda", keep_classes=None):
        self.device = device
        name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            name).to(device).eval()
        self.keep_classes = keep_classes or KEEP_CLASSES_ADE

    @torch.no_grad()
    def segment(self, img_rgb):
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        H, W = img_rgb.shape[:2]
        upsampled = F.interpolate(logits, size=(H, W),
                                  mode="bilinear", align_corners=False)
        return upsampled.argmax(dim=1)[0].cpu().numpy()

    def get_valid_mask(self, img_rgb, verbose=False):
        pred = self.segment(img_rgb)
        valid = np.isin(pred, list(self.keep_classes))
        if verbose:
            print(f"  valid ratio: {valid.mean():.2%}")
        return valid, pred