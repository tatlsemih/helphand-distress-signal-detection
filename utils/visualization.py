"""Gorsellestirme araclari.

FSM durum cubugu, bounding box, landmark ve alarm overlay cizimi.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────── Bounding Box ────────────────────────────────────

def draw_hand_bbox(
    frame:      np.ndarray,
    bbox:       Tuple[int, int, int, int],
    label:      str,
    confidence: float,
    is_distress: bool = False,
) -> np.ndarray:
    """El bounding box ve etiketini cizer."""
    x1, y1, x2, y2 = bbox
    color     = (0, 0, 255) if is_distress else (0, 210, 0)
    thickness = 3 if is_distress else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    bg_y1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────── Landmark ────────────────────────────────────────

def draw_landmarks(
    frame:       np.ndarray,
    landmarks:   np.ndarray,
    connections: Optional[list] = None,
) -> np.ndarray:
    """El landmark noktalarini cizer."""
    h, w = frame.shape[:2]
    for x, y, _ in landmarks:
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)
    if connections:
        for start, end in connections:
            sx = (int(landmarks[start][0] * w), int(landmarks[start][1] * h))
            ex = (int(landmarks[end][0] * w),   int(landmarks[end][1] * h))
            cv2.line(frame, sx, ex, (0, 200, 200), 1)
    return frame


# ─────────────────────────── FSM Durum Cubugu ────────────────────────────────

def draw_fsm_state(frame: np.ndarray, fsm_result) -> np.ndarray:
    """FSM durumunu ekranin alt bandina cizer.

    Uc daireli ilerleme cubugu gosterir:
      O-O-O  (gri = idle)
      @-O-O  (turuncu = asama1 goruldu)
      @-@-O  (sari = asama2 goruldu)
      @-@-@  (kirmizi = alarm)
    """
    if fsm_result is None:
        return frame

    h, w  = frame.shape[:2]
    color = fsm_result.state_color
    label = fsm_result.state_label
    prog  = fsm_result.phase_progress  # 0-3

    # Arka plan bandi
    bar_h   = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Asama daireleri
    step_x = [int(w * 0.25), int(w * 0.50), int(w * 0.75)]
    r      = 13
    cy     = h - bar_h // 2

    for i, sx in enumerate(step_x):
        filled = i < prog
        c = color if filled else (70, 70, 70)
        cv2.circle(frame, (sx, cy), r, c, -1)
        cv2.circle(frame, (sx, cy), r, (180, 180, 180), 1)
        cv2.putText(frame, str(i + 1), (sx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        if i < 2:
            lc = color if i + 1 < prog else (70, 70, 70)
            cv2.line(frame, (sx + r + 2, cy), (step_x[i + 1] - r - 2, cy), lc, 2)

    # Durum yazisi
    cv2.putText(frame, label, (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────── Alarm Overlay ───────────────────────────────────

def draw_alert_overlay(frame: np.ndarray) -> np.ndarray:
    """Alarm durumunda kirmizi cerceve ve uyari yazisi cizer."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), 10)
    cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 180), -1)

    msg  = "TEHLIKE ISARETI TESPIT EDILDI!"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(msg, font, 0.8, 2)
    cv2.putText(overlay, msg, ((w - tw) // 2, 40),
                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    sub = "Lutfen yardim icin yetkililere bildirin"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(overlay, sub, ((w - sw) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    return frame


# ─────────────────────────── FPS ─────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """FPS bilgisini sag ust koseye cizer."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
    x = w - tw - 10
    cv2.rectangle(frame, (x - 4, 4), (w - 4, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, th + 8), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
