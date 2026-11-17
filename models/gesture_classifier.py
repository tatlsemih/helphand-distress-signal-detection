"""Dual-input jest siniflandirma modeli.

Giris 1: Goruntu kanali  (MobileNetV3-Small, 224x224)
Giris 2: Landmark kanali (MediaPipe 21 nokta = 63 deger)

Cikis: 5 sinif logitleri
  0 = phase1_open_palm
  1 = phase2_thumb_palm
  2 = phase3_fist
  3 = normal_hand
  4 = no_signal
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES  = 5
CLASS_NAMES  = [
    "phase1_open_palm",
    "phase2_thumb_palm",
    "phase3_fist",
    "normal_hand",
    "no_signal",
]
LANDMARK_DIM = 63  # 21 nokta x 3 koordinat


class LandmarkBranch(nn.Module):
    """El landmark noktalarindan ozellik cikaran FC dali."""

    def __init__(self, input_dim: int = LANDMARK_DIM, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GestureClassifier(nn.Module):
    """5 sinifli dual-input jest siniflandirici.

    image    : (B, 3, 224, 224)
    landmarks: (B, 63)
    -> logits: (B, num_classes)
    """

    def __init__(
        self,
        num_classes:     int   = NUM_CLASSES,
        landmark_dim:    int   = LANDMARK_DIM,
        landmark_hidden: int   = 256,
        fusion_dim:      int   = 512,
        dropout:         float = 0.4,
        pretrained:      bool  = True,
    ):
        super().__init__()

        # Goruntu dali: MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.image_features = backbone.features
        self.image_avgpool  = backbone.avgpool
        image_feat_dim = 576  # MobileNetV3-Small cikis boyutu

        # Landmark dali
        self.landmark_branch = LandmarkBranch(
            input_dim=landmark_dim, hidden_dim=landmark_hidden, dropout=dropout,
        )

        # Fusion + Siniflandirici
        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + landmark_hidden, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, image: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        x        = self.image_features(image)
        x        = self.image_avgpool(x)
        img_feat = x.flatten(1)
        lm_feat  = self.landmark_branch(landmarks)
        fused    = torch.cat([img_feat, lm_feat], dim=1)
        fused    = self.fusion(fused)
        return self.classifier(fused)

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> tuple[int, float, list[float]]:
        """(predicted_class, confidence, all_probs) dondurur."""
        self.eval()
        logits = self.forward(image, landmarks)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, pred = probs.max(dim=0)
        return pred.item(), conf.item(), probs.tolist()

    def freeze_backbone(self) -> None:
        for p in self.image_features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.image_features.parameters():
            p.requires_grad = True

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
