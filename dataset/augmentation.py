"""Veri artirma (augmentation) pipeline'i.

Albumentations kutuphanesi ile guclu augmentasyon.
224x224 gorsel boyutuna gore optimize edilmistir.

Egitim augmentasyonu:
  - Rotation, flip, brightness/contrast
  - Motion blur (CCTV simulasyonu)
  - CoarseDropout (kısmi gorünürluk)
  - CCTV noise ve dusuk cozunurluk simulasyonu
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalizasyon degerleri (MobileNetV3 icin)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms(config: dict | None = None) -> A.Compose:
    """Egitim veri artirma pipeline'i.

    Args:
        config: Augmentasyon konfigurasyonu (config.yaml'dan).

    Returns:
        Albumentations Compose nesnesi.
    """
    cfg = config or {}
    rot   = cfg.get("rotation_limit", 25)
    bright = cfg.get("brightness_limit", 0.35)
    cont  = cfg.get("contrast_limit", 0.35)
    blur  = cfg.get("blur_limit", 5)
    s_min = cfg.get("scale_min", 0.75) - 1.0
    s_max = cfg.get("scale_max", 1.25) - 1.0

    return A.Compose([
        # Boyutlandir
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # Geometrik donusumler
        A.HorizontalFlip(p=0.35),
        A.Rotate(limit=rot, border_mode=0, p=0.5),
        A.Affine(
            scale=(1 + s_min, 1 + s_max),
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            shear=(-5, 5),
            p=0.4,
        ),

        # Renk/parlaklik donusumleri
        A.RandomBrightnessContrast(
            brightness_limit=bright,
            contrast_limit=cont,
            p=0.6,
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.25, hue=0.05, p=0.4
        ),
        A.ToGray(p=0.05),  # Nadiren gri tonlamali (gece kamerasi)

        # Bulaniklik (hareket + lens kalitesi)
        A.OneOf([
            A.MotionBlur(blur_limit=blur, p=1.0),
            A.GaussianBlur(blur_limit=(3, blur), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.35),

        # Gurultu (CCTV sensor gurultusu)
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
        ], p=0.3),

        # Piksellesme (dusuk cozunurluk CCTV simulasyonu)
        A.Downscale(scale_range=(0.55, 0.85), p=0.2),

        # Kısmi ortu (parmaklar/el gorsel disinda kalabilir)
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=114,
            p=0.25,
        ),

        # Kare doldur ve kes
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
            border_mode=0, value=114,
        ),
        A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),

        # ImageNet normalizasyonu + Tensor donusumu
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Dogrulama/test veri on isleme pipeline'i (augmentasyon yok).

    Args:
        image_size: Hedef gorsel boyutu.

    Returns:
        Albumentations Compose nesnesi.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
