"""Model değerlendirme modülü.

Precision, Recall, F1, Confusion Matrix hesaplama.
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.augmentation import get_val_transforms
from dataset.dataset import HandGestureDataset
from models.gesture_classifier import GestureClassifier
from utils.logger import setup_logger

logger = setup_logger("evaluator")

CLASS_NAMES = [
    "phase1_open_palm",
    "phase2_thumb_palm",
    "phase3_fist",
    "normal_hand",
    "no_signal",
]


def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """Sınıflandırma metriklerini hesaplar.

    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.

    Returns:
        Metrikler sözlüğü.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }


def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str = "logs/confusion_matrix.png"):
    """Confusion matrix görselleştirmesi oluşturur ve kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="Gercek",
        xlabel="Tahmin",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Hücre değerlerini yaz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix kaydedildi: {save_path}")


@torch.no_grad()
def evaluate(config_path: str = "config/config.yaml", model_path: str | None = None):
    """Test seti üzerinde modeli değerlendirir.

    Args:
        config_path: Konfigürasyon dosyası yolu.
        model_path: Model checkpoint yolu. None ise varsayılan kullanılır.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model yükle
    model = GestureClassifier(
        num_classes=model_cfg["num_classes"],
        landmark_dim=model_cfg["landmark_dim"],
        landmark_hidden=model_cfg["landmark_hidden"],
        fusion_dim=model_cfg["fusion_dim"],
        dropout=model_cfg["dropout"],
        pretrained=False,
    ).to(device)

    ckpt_path = model_path or str(Path(train_cfg["checkpoint_dir"]) / "best_model.pth")
    if not Path(ckpt_path).exists():
        logger.error(f"Model dosyasi bulunamadi: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model yuklendi: {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")

    # Test veri seti
    data_dir = data_cfg["data_dir"]
    test_dataset = HandGestureDataset(
        data_dir=data_dir,
        split_file=str(Path(data_dir) / "splits" / "test.txt"),
        transform=get_val_transforms(),
    )

    if len(test_dataset) == 0:
        logger.error("Test veri seti bos!")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    # Tahminler
    all_preds = []
    all_labels = []

    for batch in test_loader:
        images = batch["image"].to(device)
        landmarks = batch["landmarks"].to(device)
        labels = batch["label"]

        logits = model(images, landmarks)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Metrikler
    metrics = calculate_metrics(all_labels, all_preds)

    logger.info("\n=== DEGERLENDIRME SONUCLARI ===")
    logger.info(f"Dogruluk:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall:    {metrics['recall_weighted']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_weighted']:.4f}")

    logger.info("\nSinif bazli sonuclar:")
    for i, name in enumerate(CLASS_NAMES):
        logger.info(
            f"  {name}: P={metrics['precision_per_class'][i]:.4f} "
            f"R={metrics['recall_per_class'][i]:.4f} "
            f"F1={metrics['f1_per_class'][i]:.4f}"
        )

    # Detaylı rapor
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
    logger.info(f"\n{report}")

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds)
