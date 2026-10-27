import torch
import torch.nn as nn
import torchvision.models as tvm
from config import STAT_FEAT_DIM, IMG_EMBED_DIM


class ImageEncoder(nn.Module):
    """冻结的 ResNet18，提取 512 维图像特征"""

    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x).flatten(1)


class ScalePredictor(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.image_encoder = ImageEncoder()
        # 动态注册 Anchor 作为 buffer（不参与梯度更新，但随模型保存）
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.n_anchors = len(anchors)

        in_dim = STAT_FEAT_DIM + IMG_EMBED_DIM
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(64, self.n_anchors)
        self.res_head = nn.Linear(64, self.n_anchors)

        # 初始化残差为 0，保证刚开始训练时输出等于 anchor 本身
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    def forward(self, feat_stat, img_tensor):
        img_embed = self.image_encoder(img_tensor)
        x = torch.cat([feat_stat, img_embed], dim=-1)
        h = self.trunk(x)

        cls_logits = self.cls_head(h)  # (B, K)
        residuals = self.res_head(h)  # (B, K)
        prob = torch.softmax(cls_logits, dim=-1)  # (B, K)

        A = self.anchors.unsqueeze(0)  # (1, K)
        S = (prob * (A + residuals)).sum(dim=-1)  # (B,)
        return S, prob, cls_logits

