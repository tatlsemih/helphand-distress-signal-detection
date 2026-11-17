"""PyTorch Dataset sinifi - 5 sinifli el jest veri seti.

Her ornek:
  - 224x224 el goruntuleri (RGB)
  - 21 landmark noktasi (63 deger) - bos ise sifir
  - Sinif etiketi (0-4)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

# Sinif tanimlari
CLASS_MAP = {
    "phase1_open_palm":  0,
    "phase2_thumb_palm": 1,
    "phase3_fist":       2,
    "normal_hand":       3,
    "no_signal":         4,
}
CLASS_NAMES   = list(CLASS_MAP.keys())
VALID_EXT     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class HandGestureDataset(Dataset):
    """5 sinifli el jesti veri seti.

    Dizin yapisi:
        data/images/
            phase1_open_palm/   *.jpg
            phase2_thumb_palm/  *.jpg
            phase3_fist/        *.jpg
            normal_hand/        *.jpg
            no_signal/          *.jpg

    Split dosyasi formati (satirda): class_name/dosya_adi_uzantisiz
    """

    def __init__(
        self,
        data_dir:   str,
        split_file: str | None = None,
        transform:  A.Compose | None = None,
    ):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        image_dir = self.data_dir / "images"

        if split_file and Path(split_file).exists():
            with open(split_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts      = line.split("/")
                    class_name = parts[0]
                    filename   = parts[1]
                    if class_name not in CLASS_MAP:
                        continue
                    # Uzantisiz isme .jpg ekle
                    img_path = image_dir / class_name / f"{filename}.jpg"
                    if not img_path.exists():
                        # PNG gibi baska uzanti denemesi
                        for ext in VALID_EXT:
                            alt = image_dir / class_name / f"{filename}{ext}"
                            if alt.exists():
                                img_path = alt
                                break
                    if img_path.exists():
                        self.samples.append((img_path, CLASS_MAP[class_name]))
        else:
            # Split dosyasi yoksa tum veriyi tara
            for class_name, label in CLASS_MAP.items():
                class_dir = image_dir / class_name
                if not class_dir.exists():
                    continue
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in VALID_EXT:
                        self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]

        # Goruntu yukle (BGR -> RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.full((224, 224, 3), 114, dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentasyon
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Landmark: yok ise sifir vektor (model landmark olmadan da calismali)
        landmarks = torch.zeros(63, dtype=torch.float32)

        return {
            "image":     image,
            "landmarks": landmarks,
            "label":     torch.tensor(label, dtype=torch.long),
        }

    def class_counts(self) -> dict:
        """Sinif basi ornek sayilarini dondurur."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts
