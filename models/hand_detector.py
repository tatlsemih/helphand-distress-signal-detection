"""MediaPipe tabanlı el tespit modülü."""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass


@dataclass
class HandDetection:
    """Tek bir el tespiti sonucu."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    landmarks: np.ndarray            # (21, 3) normalize koordinatlar
    confidence: float
    handedness: str                  # "Left" veya "Right"


class HandDetector:
    """MediaPipe Hands wrapper sınıfı.

    El tespiti yapar, bounding box ve 21 landmark noktası döndürür.
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Başparmak
        (0, 5), (5, 6), (6, 7), (7, 8),       # İşaret parmağı
        (0, 9), (9, 10), (10, 11), (11, 12),  # Orta parmak
        (0, 13), (13, 14), (14, 15), (15, 16),# Yüzük parmağı
        (0, 17), (17, 18), (18, 19), (19, 20),# Serçe parmak
        (5, 9), (9, 13), (13, 17),             # Avuç içi bağlantıları
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        model_complexity: int = 1,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def detect(self, frame: np.ndarray, padding: float = 0.2) -> list[HandDetection]:
        """Frame'de el tespiti yapar.

        Args:
            frame: BGR görüntü (OpenCV formatı).
            padding: Bounding box etrafına eklenecek padding oranı.

        Returns:
            Tespit edilen ellerin listesi.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            return detections

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Landmark'ları numpy array'e dönüştür
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )

            # Bounding box hesapla
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Padding ekle
            bw = x_max - x_min
            bh = y_max - y_min
            x_min = max(0, x_min - bw * padding)
            y_min = max(0, y_min - bh * padding)
            x_max = min(1, x_max + bw * padding)
            y_max = min(1, y_max + bh * padding)

            bbox = (
                int(x_min * w),
                int(y_min * h),
                int(x_max * w),
                int(y_max * h),
            )

            confidence = handedness_info.classification[0].score
            handedness = handedness_info.classification[0].label

            detections.append(HandDetection(
                bbox=bbox,
                landmarks=landmarks,
                confidence=confidence,
                handedness=handedness,
            ))

        return detections

    def crop_hand(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Bounding box'a göre el bölgesini kırpar.

        Args:
            frame: BGR görüntü.
            bbox: (x1, y1, x2, y2) koordinatları.

        Returns:
            Kırpılmış el görüntüsü.
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2].copy()

    def close(self):
        """MediaPipe kaynaklarını serbest bırakır."""
        try:
            if hasattr(self, "hands") and self.hands is not None:
                self.hands.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
