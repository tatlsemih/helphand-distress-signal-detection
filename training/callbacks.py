"""Eğitim callback'leri.

Early stopping ve model checkpoint yönetimi.
"""

from pathlib import Path

import torch

from utils.logger import setup_logger

logger = setup_logger("callbacks")


class EarlyStopping:
    """Doğrulama metriği iyileşmediğinde eğitimi durdurur."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        """
        Args:
            patience: İyileşme olmadan beklenecek epoch sayısı.
            min_delta: Minimum iyileşme miktarı.
            mode: "max" (F1 gibi) veya "min" (loss gibi).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Yeni değeri kontrol eder.

        Returns:
            True ise eğitim durdurulmalı.
        """
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping tetiklendi. "
                    f"{self.patience} epoch boyunca iyilesme olmadi."
                )

        return self.should_stop


class ModelCheckpoint:
    """En iyi modeli kaydeder."""

    def __init__(self, save_dir: str = "checkpoints", mode: str = "max"):
        """
        Args:
            save_dir: Checkpoint kayıt dizini.
            mode: "max" veya "min".
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def step(
        self,
        value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        extra: dict | None = None,
    ) -> bool:
        """Metrik iyileştiyse modeli kaydeder.

        Returns:
            True ise model kaydedildi.
        """
        if self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_value": self.best_value,
            }
            if extra:
                checkpoint.update(extra)

            path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"Model kaydedildi: {path} (deger={value:.4f})")
            return True

        return False
