"""Gercek zamanli tespit engine'i.

Kamera -> El Tespiti -> CNN -> FSM -> Gorsellestirme -> Alarm
"""

import time

import cv2
import yaml

from inference.camera import CameraStream
from inference.alert import AlertManager
from models.distress_detector import DistressDetector
from utils.visualization import (
    draw_hand_bbox,
    draw_alert_overlay,
    draw_fps,
    draw_fsm_state,
)
from utils.logger import setup_logger

logger = setup_logger("realtime_detector")

DISTRESS_PHASES = {"phase1_open_palm", "phase2_thumb_palm", "phase3_fist"}


class RealtimeDetector:
    """Gercek zamanli tehlike isareti tespit sistemi."""

    def __init__(self, config_path: str = "config/config.yaml", model_path: str | None = None):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        inf_cfg = self.config["inference"]
        cam_cfg = self.config["camera"]
        mp_cfg  = self.config["mediapipe"]

        if model_path is None:
            model_path = f"{self.config['training']['checkpoint_dir']}/best_model.pth"

        self.detector = DistressDetector(
            model_path       = model_path,
            mediapipe_config = mp_cfg,
            model_config     = self.config["model"],
            fsm_config       = self.config,
        )

        self.alert_manager  = AlertManager(cooldown=inf_cfg["alert_cooldown"])
        self.cam_source     = cam_cfg["source"]
        self.cam_width      = cam_cfg["width"]
        self.cam_height     = cam_cfg["height"]
        self.cam_fps        = cam_cfg["fps"]
        self.display_fps    = inf_cfg.get("display_fps", True)
        self.show_fsm_state = inf_cfg.get("show_fsm_state", True)

    def run(self, source: int | str | None = None) -> None:
        """Gercek zamanli tespit dongusu.
        Tuslar: q=cikis | r=sifirla | s=ekran goruntusu
        """
        src = source if source is not None else self.cam_source
        camera = CameraStream(
            source=src, width=self.cam_width,
            height=self.cam_height, fps=self.cam_fps,
        ).start()

        logger.info(f"Tespit baslatildi | Kaynak: {src}")
        logger.info("q=cikis  r=sifirla  s=ekran_goruntusu")

        frame_times    = []
        screenshot_cnt = 0

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                t0     = time.time()
                result = self.detector.process_frame(frame)

                # Bounding box cizimi
                for pred in result.predictions:
                    is_d = pred["class_name"] in DISTRESS_PHASES
                    draw_hand_bbox(frame, pred["bbox"], pred["class_name"], pred["confidence"], is_d)

                # FSM durum cubugu
                if self.show_fsm_state and result.fsm_result is not None:
                    draw_fsm_state(frame, result.fsm_result)

                # Alarm
                if result.is_alert:
                    draw_alert_overlay(frame)
                    if result.fsm_result and result.fsm_result.just_triggered:
                        best = result.best_prediction
                        self.alert_manager.trigger(
                            confidence=best["confidence"] if best else 0.0,
                            details=f"El sayisi={len(result.hands)}",
                        )

                # FPS
                frame_times.append(time.time() - t0)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                if self.display_fps:
                    draw_fps(frame, fps)

                cv2.imshow("HelpHand - Tehlike Isareti Tespit", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.detector.reset()
                    self.alert_manager.reset()
                    logger.info("Sifirlandi.")
                elif key == ord("s"):
                    fname = f"screenshot_{screenshot_cnt:04d}.jpg"
                    cv2.imwrite(fname, frame)
                    logger.info(f"Kaydedildi: {fname}")
                    screenshot_cnt += 1

        except KeyboardInterrupt:
            logger.info("Durduruldu.")
        finally:
            camera.stop()
            self.detector.close()
            cv2.destroyAllWindows()

        logger.info(f"Toplam alarm: {self.alert_manager.alert_count}")
