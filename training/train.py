"""Model egitim modulu.

Tam egitim dongusu:
  1. Veri yukleme (5 sinif)
  2. Augmentasyon
  3. MobileNetV3 fine-tuning (backbone freeze -> unfreeze)
  4. GPU AMP ile hizli egitim
  5. Checkpoint + EarlyStopping
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from dataset.augmentation import get_train_transforms, get_val_transforms
from dataset.dataset import HandGestureDataset
from models.gesture_classifier import GestureClassifier
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.evaluate import calculate_metrics
from utils.logger import setup_logger

logger = setup_logger("trainer")


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Egitim", leave=False):
        images    = batch["image"].to(device, non_blocking=True)
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler:
            with autocast(device_type="cuda"):
                logits = model(images, landmarks)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, landmarks)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Dogrulama", leave=False):
        images    = batch["image"].to(device, non_blocking=True)
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)

        logits = model(images, landmarks)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    metrics  = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train(config_path: str = "config/config.yaml") -> None:
    """Ana egitim fonksiyonu.

    Strateji:
      Epoch 1-10  : Sadece classifier + landmark dali egitilir (backbone dondurulur)
      Epoch 11+   : Tum model egitilir (backbone cozulur, dusuk lr)
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg  = config["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cihaz: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Veri setleri ─────────────────────────────────────────────────────
    train_tf = get_train_transforms(data_cfg.get("augmentation"))
    val_tf   = get_val_transforms(model_cfg.get("image_size", 224))

    data_dir  = data_cfg["data_dir"]
    split_dir = Path(data_dir) / "splits"

    train_ds = HandGestureDataset(data_dir, str(split_dir / "train.txt"), train_tf)
    val_ds   = HandGestureDataset(data_dir, str(split_dir / "val.txt"),   val_tf)

    if len(train_ds) == 0:
        logger.error("Egitim verisi bos! Once veri hazirlayın: python main.py --mode prepare")
        return

    logger.info(f"Egitim: {len(train_ds)} | Dogrulama: {len(val_ds)}")
    logger.info("Sinif dagilimi:")
    for cls, cnt in train_ds.class_counts().items():
        logger.info(f"  {cls}: {cnt}")

    num_workers = train_cfg.get("num_workers", 4)
    batch_size  = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = GestureClassifier(
        num_classes     = model_cfg["num_classes"],
        landmark_dim    = model_cfg["landmark_dim"],
        landmark_hidden = model_cfg["landmark_hidden"],
        fusion_dim      = model_cfg["fusion_dim"],
        dropout         = model_cfg["dropout"],
        pretrained      = model_cfg["pretrained"],
    ).to(device)

    param_info = model.count_parameters()
    logger.info(
        f"Model: {param_info['total']:,} parametre "
        f"(egitilir={param_info['trainable']:,})"
    )

    # Fase 1: Backbone dondur, sadece classifier egit
    model.freeze_backbone()
    logger.info("Backbone donduruldu (ilk 10 epoch)")

    # ── Sinif agirliklari (dengesiz veri icin) ────────────────────────────
    counts = train_ds.class_counts()
    total  = sum(counts.values())
    n_cls  = model_cfg["num_classes"]
    weights = []
    for i in range(n_cls):
        from dataset.dataset import CLASS_NAMES
        c = counts.get(CLASS_NAMES[i], 1)
        weights.append(total / (n_cls * max(c, 1)))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Sinif agirliklari: {[f'{w:.2f}' for w in weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Optimizer ve Scheduler ────────────────────────────────────────────
    lr = train_cfg["learning_rate"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=lr * 0.01,
    )
    scaler  = GradScaler() if train_cfg.get("use_amp") and device.type == "cuda" else None
    if scaler:
        logger.info("AMP (Automatic Mixed Precision) aktif")

    # ── Callback'ler ──────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=train_cfg["patience"], mode="max")
    checkpoint = ModelCheckpoint(save_dir=train_cfg["checkpoint_dir"], mode="max")

    unfreeze_epoch = 10  # Bu epoch'tan itibaren backbone coz

    # ── Egitim dongusu ────────────────────────────────────────────────────
    logger.info("\nEgitim baslıyor...")
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        # Backbone unfreeze
        if epoch == unfreeze_epoch + 1:
            model.unfreeze_backbone()
            # Daha dusuk lr ile backbone'u guncelle
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr * 0.1, weight_decay=train_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=lr * 0.001,
            )
            logger.info(f"\nEpoch {epoch}: Backbone cozuldu, tam egitim basliyor (lr={lr*0.1:.6f})")

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch}/{epochs} | LR={current_lr:.6f}")

        train_loss         = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, metrics  = validate(model, val_loader, criterion, device)

        scheduler.step()

        acc  = metrics["accuracy"]
        f1   = metrics["f1_weighted"]
        prec = metrics["precision_weighted"]
        rec  = metrics["recall_weighted"]

        logger.info(
            f"  Egitim kaybi : {train_loss:.4f} | "
            f"Dogrulama kaybi: {val_loss:.4f}"
        )
        logger.info(
            f"  Accuracy={acc:.4f} | Precision={prec:.4f} | "
            f"Recall={rec:.4f} | F1={f1:.4f}"
        )

        # Sinif bazi F1 skorlari
        from dataset.dataset import CLASS_NAMES
        for i, cls in enumerate(CLASS_NAMES[:model_cfg["num_classes"]]):
            f1c = metrics["f1_per_class"][i]
            logger.info(f"    {cls}: F1={f1c:.4f}")

        checkpoint.step(f1, model, optimizer, epoch, {"metrics": metrics})

        if early_stop.step(f1):
            logger.info(f"Early stopping: {epoch}. epoch'da durduruldu.")
            break

    logger.info(f"\nEgitim tamamlandi. En iyi F1: {checkpoint.best_value:.4f}")
    logger.info(f"En iyi model: {train_cfg['checkpoint_dir']}/best_model.pth")
