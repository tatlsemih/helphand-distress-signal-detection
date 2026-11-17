"""Sonlu Durum Makinesi (Finite State Machine) - 3 Asamali Sirali Jest Tespiti.

Tehlike isaretinin 3 asamasini siraliyla algilar:
  Asama 1: Acik avuc (phase1_open_palm)
  Asama 2: Basparmak avuca (phase2_thumb_palm)
  Asama 3: Yumruk - parmaklar kapanir (phase3_fist)

Durum gecisleri:
  IDLE -> PHASE1_SEEN -> PHASE2_SEEN -> ALARM

Her gecis icin:
  - Guven esigi kontrolu (phase1 daha dusuk esik)
  - Art arda frame kararlilik kontrolu
  - Zaman asimi kontrolu
  - Bounding box konumsal tutarlilik kontrolu
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger("gesture_fsm")


class FSMState(Enum):
    """FSM durum tanimlari."""
    IDLE         = auto()   # Bekleme - hicbir asama gorulmedi
    PHASE1_SEEN  = auto()   # Asama 1 dogrulandi - acik avuc
    PHASE2_SEEN  = auto()   # Asama 2 dogrulandi - basparmak avuca
    ALARM        = auto()   # Tehlike isareti tamamlandi!


# Sinif indeksleri (model cikisi ile eslesmeli)
CLASS_PHASE1 = 0  # phase1_open_palm
CLASS_PHASE2 = 1  # phase2_thumb_palm
CLASS_PHASE3 = 2  # phase3_fist
CLASS_NORMAL = 3  # normal_hand
CLASS_NOSIG  = 4  # no_signal

# Durum renkleri (BGR - OpenCV icin)
STATE_COLORS = {
    FSMState.IDLE:        (100, 100, 100),   # Gri
    FSMState.PHASE1_SEEN: (0, 165, 255),     # Turuncu
    FSMState.PHASE2_SEEN: (0, 200, 255),     # Sari
    FSMState.ALARM:       (0, 0, 255),       # Kirmizi
}

STATE_LABELS = {
    FSMState.IDLE:        "Bekleniyor...",
    FSMState.PHASE1_SEEN: "Asama 1/3: Avuc tespit edildi",
    FSMState.PHASE2_SEEN: "Asama 2/3: Basparmak tespit edildi",
    FSMState.ALARM:       "TEHLIKE ISARETI TESPIT EDILDI!",
}


@dataclass
class FSMConfig:
    """FSM konfigurasyonu."""
    phase1_confidence: float = 0.65     # Asama 1 guven esigi (dusuk - az veri)
    phase2_confidence: float = 0.78     # Asama 2 guven esigi
    phase3_confidence: float = 0.78     # Asama 3 guven esigi
    min_stable_frames: int   = 3        # Art arda kac frame gorulmeli
    phase_timeout: float     = 6.0      # Sonraki asamanin gelmesi icin max sure
    total_timeout: float     = 15.0     # Toplam hareket icin max sure
    bbox_iou_threshold: float = 0.25    # Konumsal tutarlilik icin min IoU
    alarm_hold_duration: float = 3.0    # Alarm ekranda kac saniye gorunsun

    @classmethod
    def from_config(cls, cfg: dict) -> "FSMConfig":
        """config.yaml'dan FSMConfig olusturur."""
        fsm_cfg = cfg.get("fsm", {})
        return cls(
            phase1_confidence  = fsm_cfg.get("phase1_confidence", 0.65),
            phase2_confidence  = fsm_cfg.get("phase2_confidence", 0.78),
            phase3_confidence  = fsm_cfg.get("phase3_confidence", 0.78),
            min_stable_frames  = fsm_cfg.get("min_stable_frames", 3),
            phase_timeout      = fsm_cfg.get("phase_timeout", 6.0),
            total_timeout      = fsm_cfg.get("total_timeout", 15.0),
            bbox_iou_threshold = fsm_cfg.get("bbox_iou_threshold", 0.25),
        )


@dataclass
class FSMResult:
    """Tek bir frame icin FSM sonucu."""
    state: FSMState                    # Mevcut durum
    state_label: str                   # Durum aciklamasi
    state_color: Tuple[int, int, int]  # BGR renk
    is_alarm: bool                     # Alarm tetiklendi mi
    phase_progress: int                # Kac asama tamamlandi (0-3)
    stable_frame_count: int            # Mevcut asamada art arda frame sayisi
    time_in_state: float               # Mevcut durumda gecen sure
    just_triggered: bool               # Bu frame'de alarm ilk kez mi tetiklendi


class GestureFSM:
    """3 asamali tehlike jest sonlu durum makinesi.

    Kullanim:
        fsm = GestureFSM(FSMConfig.from_config(yaml_config))

        # Her frame'de:
        result = fsm.update(
            predicted_class=2,
            confidence=0.82,
            bbox=(100, 50, 300, 400)
        )

        if result.is_alarm:
            trigger_alert()
    """

    def __init__(self, config: Optional[FSMConfig] = None):
        self.cfg = config or FSMConfig()
        self._reset_internal()
        logger.info(
            f"FSM baslatildi | "
            f"Esikler: P1={self.cfg.phase1_confidence:.2f} "
            f"P2={self.cfg.phase2_confidence:.2f} "
            f"P3={self.cfg.phase3_confidence:.2f} | "
            f"Min frame={self.cfg.min_stable_frames} | "
            f"Timeout={self.cfg.phase_timeout}s"
        )

    def _reset_internal(self) -> None:
        """Tum ic durumu sifirlar."""
        self._state             = FSMState.IDLE
        self._stable_count      = 0          # Mevcut asamada art arda dogru frame
        self._state_start_time  = 0.0        # Mevcut duruma giris zamani
        self._sequence_start    = 0.0        # Asama 1'in ilk gorulme zamani
        self._last_bbox         = None       # Son gecerli bounding box
        self._alarm_time        = 0.0        # Alarm tetiklenme zamani
        self._just_triggered    = False
        self._nosig_count       = 0          # Ardarda no_signal frame sayisi (grace period)

    @property
    def state(self) -> FSMState:
        return self._state

    @property
    def phase_progress(self) -> int:
        """Tamamlanan asama sayisi (0-3)."""
        return {
            FSMState.IDLE:        0,
            FSMState.PHASE1_SEEN: 1,
            FSMState.PHASE2_SEEN: 2,
            FSMState.ALARM:       3,
        }[self._state]

    def reset(self) -> None:
        """FSM'i IDLE'a sifirlar ve baslangic durumuna doner."""
        prev = self._state
        self._reset_internal()
        if prev != FSMState.IDLE:
            logger.info(f"FSM sifirlandi ({prev.name} -> IDLE)")

    def update(
        self,
        predicted_class: int,
        confidence: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> FSMResult:
        """Yeni frame tahmini ile FSM'i gunceller.

        Args:
            predicted_class: Model cikisi (0-4 sinif indeksi).
            confidence     : Model guven skoru (0.0-1.0).
            bbox           : El bounding box'i (x1, y1, x2, y2).

        Returns:
            FSMResult nesnesi.
        """
        self._just_triggered = False
        now = time.time()

        # ─── ALARM hold suresi dolduysa IDLE'a don ───────────────────────
        if self._state == FSMState.ALARM:
            if now - self._alarm_time >= self.cfg.alarm_hold_duration:
                self.reset()
            return self._build_result(now)

        # ─── Toplam zaman asimi kontrolu ─────────────────────────────────
        if (self._state != FSMState.IDLE
                and (now - self._sequence_start) > self.cfg.total_timeout):
            logger.info(f"Toplam zaman asimi ({self.cfg.total_timeout}s) -> IDLE")
            self.reset()
            return self._build_result(now)

        # ─── Asama gecisi zaman asimi kontrolu ───────────────────────────
        if (self._state != FSMState.IDLE
                and (now - self._state_start_time) > self.cfg.phase_timeout):
            logger.info(
                f"Asama zaman asimi ({self.cfg.phase_timeout}s) "
                f"[{self._state.name}] -> IDLE"
            )
            self.reset()
            return self._build_result(now)

        # ─── Durum gecis mantigi ──────────────────────────────────────────
        if self._state == FSMState.IDLE:
            self._handle_idle(predicted_class, confidence, bbox, now)

        elif self._state == FSMState.PHASE1_SEEN:
            self._handle_phase1(predicted_class, confidence, bbox, now)

        elif self._state == FSMState.PHASE2_SEEN:
            self._handle_phase2(predicted_class, confidence, bbox, now)

        return self._build_result(now)

    # ─────────────────────── Durum isleyicileri ──────────────────────────

    def _handle_idle(
        self,
        cls: int,
        conf: float,
        bbox: Optional[Tuple],
        now: float,
    ) -> None:
        """IDLE durumunda phase1 arayisinda."""
        if cls == CLASS_PHASE1 and conf >= self.cfg.phase1_confidence:
            self._stable_count += 1
            if self._stable_count >= self.cfg.min_stable_frames:
                self._transition_to(FSMState.PHASE1_SEEN, bbox, now)
                self._sequence_start = now
                logger.info(
                    f"ASAMA 1 onaylandi (conf={conf:.2f}, "
                    f"frame={self._stable_count}) -> PHASE1_SEEN"
                )
        else:
            # Diger siniflarda stable sayacini sifirla
            if cls != CLASS_PHASE1:
                self._stable_count = 0

    def _handle_phase1(
        self,
        cls: int,
        conf: float,
        bbox: Optional[Tuple],
        now: float,
    ) -> None:
        """PHASE1_SEEN durumunda phase2 arayisinda."""
        if cls == CLASS_PHASE2 and conf >= self.cfg.phase2_confidence:
            if bbox is not None and self._last_bbox is not None:
                iou = _compute_iou(bbox, self._last_bbox)
                if iou < self.cfg.bbox_iou_threshold:
                    # Farkli konumda el - muhtemelen baska bir el, say
                    self._stable_count = 0
                    return

            self._stable_count += 1
            if self._stable_count >= self.cfg.min_stable_frames:
                self._transition_to(FSMState.PHASE2_SEEN, bbox, now)
                logger.info(
                    f"ASAMA 2 onaylandi (conf={conf:.2f}) -> PHASE2_SEEN"
                )
        elif cls in (CLASS_NORMAL, CLASS_NOSIG):
            # Normal hareket gorulunce sifirla
            self._stable_count = 0
        # Phase1 tekrar gorulurse (geri gitmis) - stabil olarak bekle

    def _handle_phase2(
        self,
        cls: int,
        conf: float,
        bbox: Optional[Tuple],
        now: float,
    ) -> None:
        """PHASE2_SEEN durumunda phase3 arayisinda.

        Not: Yumruk yaparken MediaPipe bazen dusuk guven verir veya
        phase2 tekrar gorur. Bu nedenle:
        - IoU kontrolu daha esnek (config'den gelir, varsayilan 0.10)
        - Sadece normal_hand / no_signal stable_count'u sifirlar
        - phase1/phase2 tekrar gorulurse beklemeye devam edilir
        """
        if cls == CLASS_PHASE3 and conf >= self.cfg.phase3_confidence:
            # Konum tutarliligi: yumrukta bbox kuculebilir, esnek kontrol
            if bbox is not None and self._last_bbox is not None:
                iou = _compute_iou(bbox, self._last_bbox)
                if iou < self.cfg.bbox_iou_threshold:
                    # Tamamen farkli konumda el - baska bir el olabilir
                    logger.debug(f"Phase3 IoU dusuk ({iou:.2f}) - atlanıyor")
                    self._stable_count = 0
                    return

            self._stable_count += 1
            logger.debug(
                f"Phase3 onay: cls={cls} conf={conf:.2f} "
                f"stable={self._stable_count}/{self.cfg.min_stable_frames}"
            )
            if self._stable_count >= self.cfg.min_stable_frames:
                self._transition_to(FSMState.ALARM, bbox, now)
                self._alarm_time     = now
                self._just_triggered = True
                logger.warning(
                    f"!!! TEHLIKE ISARETI TAMAMLANDI !!! "
                    f"(conf={conf:.2f}) -> ALARM"
                )
        elif cls == CLASS_NORMAL:
            # Normal el gorunce sifirla (kullanici bilinçli baska hareket yapiyor)
            if self._stable_count > 0:
                logger.debug(f"Phase3 stable_count sifirlandi (normal_hand)")
            self._stable_count = 0
        elif cls == CLASS_NOSIG:
            # El kaybolunca (MediaPipe kaybetti) grace period uygula:
            # Yumruk yaparken MediaPipe eli geçici kaybedebilir.
            # Ardarda 8+ no_signal gelirse sifirla, aksi halde bekle.
            self._nosig_count = getattr(self, "_nosig_count", 0) + 1
            if self._nosig_count >= 8:
                logger.debug(f"Phase3: {self._nosig_count} frame no_signal -> sifirla")
                self._stable_count = 0
                self._nosig_count  = 0
            # else: kısa kayıplar görmezden gel
        else:
            self._nosig_count = 0   # no_signal sayacini sifirla
        # phase1, phase2 tekrar gorulurse (gecis animasyonu) - sayaci koru

    # ─────────────────────── Yardimci metodlar ───────────────────────────

    def _transition_to(
        self,
        new_state: FSMState,
        bbox: Optional[Tuple],
        now: float,
    ) -> None:
        """Yeni duruma gecis yapar."""
        self._state            = new_state
        self._stable_count     = 0
        self._state_start_time = now
        self._last_bbox        = bbox

    def _build_result(self, now: float) -> FSMResult:
        """Mevcut durumdan FSMResult olusturur."""
        time_in_state = now - self._state_start_time if self._state_start_time else 0.0
        return FSMResult(
            state              = self._state,
            state_label        = STATE_LABELS[self._state],
            state_color        = STATE_COLORS[self._state],
            is_alarm           = (self._state == FSMState.ALARM),
            phase_progress     = self.phase_progress,
            stable_frame_count = self._stable_count,
            time_in_state      = time_in_state,
            just_triggered     = self._just_triggered,
        )

    def get_debug_info(self) -> dict:
        """Debug icin ic durum bilgisi."""
        return {
            "state":         self._state.name,
            "stable_count":  self._stable_count,
            "phase_progress": self.phase_progress,
            "time_in_state": time.time() - self._state_start_time
                             if self._state_start_time else 0.0,
            "total_elapsed": time.time() - self._sequence_start
                             if self._sequence_start else 0.0,
        }


# ──────────────────────── Yardimci fonksiyonlar ──────────────────────────

def _compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Iki bounding box arasindaki Intersection over Union (IoU) hesaplar.

    Args:
        box1, box2: (x1, y1, x2, y2) koordinatlari.

    Returns:
        IoU degeri (0.0 - 1.0).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
