"""Veri toplama aracı.

Webcam üzerinden interaktif veri toplama yapar.
Kullanıcıya hangi jesti yapması gerektiğini gösterir,
otomatik olarak el görüntüsünü kırpar ve landmark'ları kaydeder.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from models.hand_detector import HandDetector


class DataCollector:
    """Webcam tabanlı veri toplama aracı."""

    CLASSES = ["distress_signal", "normal_hand", "no_signal"]

    INSTRUCTIONS = {
        "distress_signal": [
            "TEHLIKE ISARETI - Asama 1: Avuc ici kameraya donuk, acik el",
            "TEHLIKE ISARETI - Asama 2: Basparmak avuc icine katlanmis",
            "TEHLIKE ISARETI - Asama 3: Parmaklar basparmak uzerine kapanmis",
        ],
        "normal_hand": [
            "NORMAL EL - Acik el (selam verir gibi)",
            "NORMAL EL - Yumruk",
            "NORMAL EL - Isaret parmagi yukari",
            "NORMAL EL - OK isareti",
            "NORMAL EL - Baris isareti (V)",
        ],
        "no_signal": [
            "ISARET YOK - Elinizi kameradan uzaklastirin",
        ],
    }

    def __init__(
        self,
        output_dir: str = "data",
        camera_source: int = 0,
        image_size: tuple[int, int] = (640, 480),
    ):
        self.output_dir = Path(output_dir)
        self.camera_source = camera_source
        self.image_size = image_size
        self.detector = HandDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Dizinleri oluştur
        for cls in self.CLASSES:
            (self.output_dir / "images" / cls).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "landmarks" / cls).mkdir(parents=True, exist_ok=True)

    def collect(self, target_class: str, num_samples: int = 100):
        """Belirli bir sınıf için veri toplar.

        Args:
            target_class: Hedef sınıf adı.
            num_samples: Toplanacak örnek sayısı.
        """
        if target_class not in self.CLASSES:
            raise ValueError(f"Gecersiz sinif: {target_class}. Gecerli siniflar: {self.CLASSES}")

        cap = cv2.VideoCapture(self.camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])

        if not cap.isOpened():
            raise RuntimeError("Kamera acilamadi!")

        instructions = self.INSTRUCTIONS[target_class]
        collected = self._count_existing(target_class)
        instruction_idx = 0
        last_save_time = 0
        save_interval = 0.3  # Saniye cinsinden minimum kayıt aralığı

        print(f"\n--- {target_class} sinifi icin veri toplama ---")
        print(f"Mevcut: {collected}, Hedef: {num_samples}")
        print("SPACE: Kaydet | N: Sonraki talimat | Q: Cikis\n")

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # Talimat göster
            instruction = instructions[instruction_idx % len(instructions)]
            cv2.putText(
                display, instruction, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )
            cv2.putText(
                display, f"Toplanan: {collected}/{num_samples}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # El tespiti
            detections = self.detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Veri Toplama", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                instruction_idx += 1
            elif key == ord(" "):
                now = time.time()
                if now - last_save_time < save_interval:
                    continue

                if target_class == "no_signal":
                    # El yokken kaydet (boş görüntü)
                    self._save_sample(frame, None, target_class, collected)
                    collected += 1
                    last_save_time = now
                elif detections:
                    for det in detections:
                        hand_crop = self.detector.crop_hand(frame, det.bbox)
                        if hand_crop.size > 0:
                            self._save_sample(hand_crop, det.landmarks, target_class, collected)
                            collected += 1
                            last_save_time = now

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nToplam {collected} ornek toplandi.")

    def _save_sample(
        self,
        image: np.ndarray,
        landmarks: np.ndarray | None,
        class_name: str,
        index: int,
    ):
        """Tek bir örneği kaydeder."""
        timestamp = int(time.time() * 1000)
        filename = f"{class_name}_{index:05d}_{timestamp}"

        img_path = self.output_dir / "images" / class_name / f"{filename}.jpg"
        cv2.imwrite(str(img_path), image)

        lm_path = self.output_dir / "landmarks" / class_name / f"{filename}.npy"
        if landmarks is not None:
            np.save(str(lm_path), landmarks)
        else:
            np.save(str(lm_path), np.zeros((21, 3), dtype=np.float32))

    def _count_existing(self, class_name: str) -> int:
        """Mevcut örnek sayısını döndürür."""
        img_dir = self.output_dir / "images" / class_name
        if not img_dir.exists():
            return 0
        return len(list(img_dir.glob("*.jpg")))

    def collect_all(self, samples_per_class: int = 100):
        """Tüm sınıflar için sırayla veri toplar."""
        for cls in self.CLASSES:
            print(f"\n{'='*50}")
            print(f"Sinif: {cls}")
            print(f"{'='*50}")
            self.collect(cls, samples_per_class)

    def close(self):
        """Kaynakları serbest bırakır."""
        self.detector.close()
