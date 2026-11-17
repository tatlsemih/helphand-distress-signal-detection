"""Veri seti hazirlama, import ve bolme modulu.

D:/Verisetleri/ altindaki gorselleri projeye aktarir,
224x224'e normalize eder ve train/val/test olarak boler.
"""

import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.logger import setup_logger

logger = setup_logger("data_prepare")

# Hand_Help_Signal klasorunu dinamik olarak tara
# Turkce karakter sorununu onlemek icin os.listdir kullanilir
def _build_source_map() -> dict:
    """D:/Verisetleri altindaki gercek klasor isimlerini bulur."""
    import os
    result = {}
    base = "D:/Verisetleri/Hand_Help_Signal"
    if os.path.exists(base):
        for d in os.listdir(base):
            full = os.path.join(base, d)
            if not os.path.isdir(full):
                continue
            d_lower = d.lower()
            # Asama 1: avuc ici / palm
            if ("avuc" in d_lower or "palm" in d_lower or "avu" in d_lower) \
               and "ba" not in d_lower and "parmak" not in d_lower:
                result[full] = "phase1_open_palm"
            # Asama 2: basparmak / thumb
            elif "ba" in d_lower and ("parmak" in d_lower or "thumb" in d_lower):
                result[full] = "phase2_thumb_palm"
            elif "thumb" in d_lower:
                result[full] = "phase2_thumb_palm"
            # Asama 3: yumruk / fist
            elif "yumruk" in d_lower or "fist" in d_lower:
                result[full] = "phase3_fist"
    # Sabit yollar
    result["D:/Verisetleri/Normal_hand"] = "normal_hand"
    result["D:/Verisetleri/No_signal"]   = "no_signal"
    return result

SOURCE_MAP = _build_source_map()

# 5 sinif listesi
CLASS_NAMES = [
    "phase1_open_palm",
    "phase2_thumb_palm",
    "phase3_fist",
    "normal_hand",
    "no_signal",
]

IMAGE_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_dataset(data_dir: str = "data", image_size: int = IMAGE_SIZE) -> dict:
    """D:/Verisetleri klasorunden gorselleri projeye aktarir.

    Her gorseli:
      - image_size x image_size boyutuna getirir (aspect ratio koruyarak)
      - data/images/<sinif>/ altina JPG olarak kaydeder

    Args:
        data_dir: Proje veri dizini.
        image_size: Hedef gorsel boyutu (kare).

    Returns:
        Sinif basi gorsel sayilarini gosteren sozluk.
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"

    counts = {cls: 0 for cls in CLASS_NAMES}

    for src_path_str, class_name in SOURCE_MAP.items():
        src_path = Path(src_path_str)
        if not src_path.exists():
            logger.warning(f"Kaynak klasor bulunamadi, atlaniyor: {src_path}")
            continue

        dst_dir = image_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = [
            f for f in src_path.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ]

        logger.info(f"{src_path.name} -> {class_name}: {len(files)} gorsel isleniyor...")

        for idx, src_file in enumerate(tqdm(files, desc=f"  {class_name}", leave=False)):
            dst_file = dst_dir / f"{class_name}_{idx:05d}.jpg"

            # Zaten varsa atla
            if dst_file.exists():
                counts[class_name] += 1
                continue

            # OpenCV Turkce/Unicode yol sorunu icin numpy ile oku
            try:
                import numpy as np
                raw = np.fromfile(str(src_file), dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            except Exception:
                img = cv2.imread(str(src_file))
            if img is None:
                logger.warning(f"Okunamadi: {src_file}")
                continue

            # RGBA -> RGB donusumu
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Aspect ratio koruyarak yeniden boyutlandir, sonra kare yapistir
            h, w = img.shape[:2]
            scale = image_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Ortalanmis kare uret
            canvas = cv2.copyMakeBorder(
                resized,
                top=(image_size - new_h) // 2,
                bottom=(image_size - new_h + 1) // 2,
                left=(image_size - new_w) // 2,
                right=(image_size - new_w + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            canvas = cv2.resize(canvas, (image_size, image_size))
            cv2.imwrite(str(dst_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[class_name] += 1

    logger.info("\nImport tamamlandi:")
    for cls, cnt in counts.items():
        logger.info(f"  {cls}: {cnt} gorsel")

    return counts


def prepare_splits(
    data_dir: str = "data",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> dict:
    """Veri setini train/val/test olarak boler.

    Once import_dataset() cagrilmis olmalidir.

    Args:
        data_dir: Veri dizini koku.
        val_split: Dogrulama seti orani.
        test_split: Test seti orani.
        seed: Rastgele tohum.

    Returns:
        Split sayilarini gosteren sozluk.
    """
    random.seed(seed)
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    splits_dir = data_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for class_name in CLASS_NAMES:
        class_dir = image_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Sinif dizini bulunamadi: {class_dir}")
            continue

        images = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        )

        for img in images:
            all_samples.append(f"{class_name}/{img.stem}")

        logger.info(f"  {class_name}: {len(images)} ornek")

    if not all_samples:
        logger.error("Hic veri bulunamadi! Once import_dataset() calistirin.")
        return {}

    random.shuffle(all_samples)
    total = len(all_samples)
    val_count = int(total * val_split)
    test_count = int(total * test_split)
    train_count = total - val_count - test_count

    train_samples = all_samples[:train_count]
    val_samples   = all_samples[train_count:train_count + val_count]
    test_samples  = all_samples[train_count + val_count:]

    for split_name, samples in [
        ("train", train_samples),
        ("val",   val_samples),
        ("test",  test_samples),
    ]:
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(s + "\n")
        logger.info(f"{split_name}: {len(samples)} ornek -> {split_file}")

    logger.info(
        f"\nToplam: {total} ornek | "
        f"Egitim={train_count} | "
        f"Dogrulama={val_count} | "
        f"Test={test_count}"
    )

    return {"train": train_count, "val": val_count, "test": test_count, "total": total}


def run_full_preparation(data_dir: str = "data") -> None:
    """Tam hazırlama: import + split.

    Tek fonksiyon ile tum hazirlik asamalarini calistirir.
    """
    logger.info("=" * 55)
    logger.info("  VERI HAZIRLAMA BASLADI")
    logger.info("=" * 55)

    logger.info("\n[1/2] Gorseller import ediliyor...")
    counts = import_dataset(data_dir=data_dir)

    total = sum(counts.values())
    if total == 0:
        logger.error("Hic gorsel aktarilamadi. Kaynak yollarini kontrol edin.")
        return

    logger.info("\n[2/2] Train/val/test split hazirlaniyor...")
    splits = prepare_splits(data_dir=data_dir)

    logger.info("\n" + "=" * 55)
    logger.info("  HAZIRLIK TAMAMLANDI")
    logger.info("=" * 55)
    logger.info(f"  Toplam gorsel : {total}")
    logger.info(f"  Egitim        : {splits.get('train', 0)}")
    logger.info(f"  Dogrulama     : {splits.get('val', 0)}")
    logger.info(f"  Test          : {splits.get('test', 0)}")
    logger.info("=" * 55)
