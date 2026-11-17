"""Ana tehlike isareti tespit pipeline'i.

MediaPipe el tespiti + CNN siniflandirmasi + FSM eslestirmesi.

Akis:
  Frame -> MediaPipe (el bul) -> Goruntu+Landmark -> CNN + Geometrik Kontrol -> FSM -> Sonuc

Geometrik Yumruk Tespiti:
  CNN modeli webcam görüntülerini yanlış sınıflandırabilir (domain shift).
  Bu nedenle MediaPipe landmark geometrisinden bağımsız yumruk tespiti eklendi:
  Tüm parmak uçları avuca katlandığında → phase3_fist olarak işaretlenir.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T

from models.hand_detector import HandDetector, HandDetection
from models.gesture_classifier import GestureClassifier, CLASS_NAMES
from models.gesture_fsm import GestureFSM, FSMConfig, FSMResult


# ─────────────────── Geometrik Yumruk Tespiti ───────────────────────────────

def _finger_curl_score(landmarks: np.ndarray) -> float:
    """MediaPipe landmark koordinatlarindan parmak kapanma skoru hesaplar.

    21 noktalı MediaPipe el modelinde:
      - Parmak ucu (tip)  : [8, 12, 16, 20]
      - PIP eklemi (orta) : [6, 10, 14, 18]
      - MCP eklemi (kök)  : [5,  9, 13, 17]

    Her parmak için: uç noktası kök ekleminden daha aşağıdaysa (y büyükse)
    parmak katlanmış sayılır.

    Returns:
        0.0 (hiç parmak katlanmamış) - 1.0 (tüm parmaklar tam katlanmış)
    """
    # Parmak ucu ve MCP çiftleri: (tip_idx, mcp_idx)
    finger_pairs = [
        (8,  5),   # İşaret
        (12, 9),   # Orta
        (16, 13),  # Yüzük
        (20, 17),  # Serçe
    ]
    curled = 0
    for tip_i, mcp_i in finger_pairs:
        tip_y = landmarks[tip_i][1]
        mcp_y = landmarks[mcp_i][1]
        pip_i = tip_i - 2          # PIP = tip - 2 (MediaPipe sıralaması)
        pip_y = landmarks[pip_i][1]
        # Uç PIP'ten ve MCP'den aşağıdaysa parmak katlanmış
        if tip_y > pip_y and tip_y > mcp_y:
            curled += 1
    return curled / 4.0


def _is_geometric_fist(
    landmarks: np.ndarray,
    curl_threshold: float = 0.75,
) -> tuple[bool, float]:
    """Landmark geometrisinden yumruk tespiti yapar.

    Args:
        landmarks: (21, 3) MediaPipe landmark koordinatları (normalize 0-1).
        curl_threshold: Yumruk sayılmak için minimum parmak kapanma oranı.

    Returns:
        (is_fist, score) - yumruk mu ve güven skoru.
    """
    score = _finger_curl_score(landmarks)
    return score >= curl_threshold, score


@dataclass
class DetectionResult:
    """Tek bir frame icin tam tespit sonucu."""
    hands:        list[HandDetection] = field(default_factory=list)
    predictions:  list[dict]          = field(default_factory=list)
    fsm_result:   FSMResult           = None
    is_alert:     bool                = False
    timestamp:    float               = 0.0

    @property
    def best_prediction(self) -> dict | None:
        """En yuksek guvenli tahmini dondurur."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p["confidence"])


class DistressDetector:
    """Tehlike el isareti tespit sistemi.

    MediaPipe ile el tespiti, CNN ile siniflandirma,
    FSM ile 3 asamali sirali dogrulama yapar.
    """

    def __init__(
        self,
        model_path:         str | None = None,
        device:             str        = "cuda",
        mediapipe_config:   dict | None = None,
        model_config:       dict | None = None,
        fsm_config:         dict | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # El dedektoru (MediaPipe)
        mp_cfg = mediapipe_config or {}
        self.hand_detector = HandDetector(
            min_detection_confidence = mp_cfg.get("min_detection_confidence", 0.65),
            min_tracking_confidence  = mp_cfg.get("min_tracking_confidence", 0.50),
            max_num_hands            = mp_cfg.get("max_num_hands", 2),
            model_complexity         = mp_cfg.get("model_complexity", 1),
        )

        # Jest siniflandirici (CNN)
        m_cfg = model_config or {}
        self.classifier = GestureClassifier(
            num_classes     = m_cfg.get("num_classes", 5),
            landmark_dim    = m_cfg.get("landmark_dim", 63),
            landmark_hidden = m_cfg.get("landmark_hidden", 256),
            fusion_dim      = m_cfg.get("fusion_dim", 512),
            dropout         = m_cfg.get("dropout", 0.4),
            pretrained      = False,
        ).to(self.device)

        # Model agirliklarini yukle
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.eval()

        # FSM
        fsm_cfg_obj = FSMConfig.from_config(fsm_config or {}) if fsm_config else FSMConfig()
        self.fsm = GestureFSM(config=fsm_cfg_obj)

        # Goruntu on isleme
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Tek bir frame'i isler ve FSM sonucunu dondurur.

        Args:
            frame: BGR goruntu (OpenCV).

        Returns:
            DetectionResult nesnesi.
        """
        result = DetectionResult(timestamp=time.time())

        # El tespiti
        detections = self.hand_detector.detect(frame)
        result.hands = detections

        if not detections:
            # El kayip: Eger PHASE2_SEEN durumundaysak FSM'i guncelleme.
            # Yumruk yaparken MediaPipe geçici olarak eli kaybedebilir (10-20 frame).
            # FSM'deki phase_timeout (6s) bu durumu zaten handle eder.
            # Sadece IDLE veya PHASE1_SEEN'de no_signal gonder.
            from models.gesture_fsm import FSMState
            if self.fsm.state == FSMState.PHASE2_SEEN:
                # Gecici kayip: mevcut FSM sonucunu don, guncelleme
                fsm_result = self.fsm._build_result(time.time())
                result.fsm_result = fsm_result
                result.is_alert   = fsm_result.is_alarm
                return result
            # Diger durumlarda no_signal bildir
            fsm_result = self.fsm.update(
                predicted_class=4,  # no_signal
                confidence=1.0,
                bbox=None,
            )
            result.fsm_result = fsm_result
            result.is_alert   = fsm_result.is_alarm
            return result

        # En buyuk eli al (bounding box alani en buyuk)
        primary = max(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )

        # El bolgesini kirp ve siniflandir
        hand_crop = self.hand_detector.crop_hand(frame, primary.bbox)
        if hand_crop.size == 0 or min(hand_crop.shape[:2]) < 20:
            # Kirpma basarisiz - PHASE2_SEEN ise gecici kayip say
            from models.gesture_fsm import FSMState
            if self.fsm.state == FSMState.PHASE2_SEEN:
                fsm_result = self.fsm._build_result(time.time())
            else:
                fsm_result = self.fsm.update(predicted_class=4, confidence=1.0, bbox=None)
            result.fsm_result = fsm_result
            result.is_alert   = fsm_result.is_alarm
            return result

        # BGR -> RGB ve on isleme
        hand_rgb    = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        img_tensor  = self.transform(hand_rgb).unsqueeze(0).to(self.device)
        lm_flat     = primary.landmarks.flatten()
        lm_tensor   = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(self.device)

        # CNN Siniflandirma
        pred_cls, confidence, all_probs = self.classifier.predict(img_tensor, lm_tensor)
        class_name = CLASS_NAMES[pred_cls]

        # ── Geometrik Yumruk Tespiti (Domain Shift Korumasi) ─────────────────
        # CNN modeli egitim verisine cok bagli olabilir. Landmark geometrisinden
        # bagimsiz yumruk tespiti yapilir ve CNN ile birlestirilir.
        geo_fist, geo_score = _is_geometric_fist(primary.landmarks, curl_threshold=0.75)
        if geo_fist:
            # Parmaklar geometrik olarak katlanmis: phase3_fist olarak isle
            # CNN phase3 diyorsa: max(cnn_conf, geo_score) al
            # CNN baska sey diyorsa: geometrik skoru kullan (min 0.72)
            if pred_cls == 2:  # phase3_fist zaten
                confidence = max(confidence, geo_score)
            else:
                # CNN yanilıyor - landmark geometrisi daha guvenilir
                pred_cls   = 2    # phase3_fist
                confidence = max(0.72, geo_score)
                class_name = CLASS_NAMES[2]
        # ─────────────────────────────────────────────────────────────────────

        result.predictions.append({
            "class":      pred_cls,
            "class_name": class_name,
            "confidence": confidence,
            "all_probs":  all_probs,
            "bbox":       primary.bbox,
            "handedness": primary.handedness,
            "geo_fist":   geo_fist,
            "geo_score":  geo_score,
        })

        # Diger eller icin de tahmin yap (goruntuleme icin)
        for det in detections:
            if det is primary:
                continue
            crop = self.hand_detector.crop_hand(frame, det.bbox)
            if crop.size == 0 or min(crop.shape[:2]) < 20:
                continue
            rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            it    = self.transform(rgb).unsqueeze(0).to(self.device)
            lt    = torch.tensor(det.landmarks.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            pc, cf, ap = self.classifier.predict(it, lt)
            # Diger el icin de geometrik kontrol
            gf, gs = _is_geometric_fist(det.landmarks)
            if gf and pc != 2:
                pc, cf = 2, max(0.72, gs)
            result.predictions.append({
                "class": pc, "class_name": CLASS_NAMES[pc],
                "confidence": cf, "all_probs": ap,
                "bbox": det.bbox, "handedness": det.handedness,
            })

        # FSM'i guncelle (birincil el ile)
        fsm_result = self.fsm.update(
            predicted_class = pred_cls,
            confidence      = confidence,
            bbox            = primary.bbox,
        )

        result.fsm_result = fsm_result
        result.is_alert   = fsm_result.is_alarm
        return result

    def reset(self) -> None:
        """FSM ve dedektoru sifirlar."""
        self.fsm.reset()

    def close(self) -> None:
        """Kaynaklari serbest birakir."""
        self.hand_detector.close()
