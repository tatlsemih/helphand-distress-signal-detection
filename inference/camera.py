"""Kamera ve video stream yönetimi.

Webcam, RTSP ve video dosyası desteği.
Threading ile asenkron frame okuma.
"""

import time
from threading import Thread, Lock
from queue import Queue

import cv2
import numpy as np


class CameraStream:
    """Asenkron kamera/video stream okuyucu.

    Ayrı bir thread'de frame okur, ana thread'i bloklamaz.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2,
    ):
        """
        Args:
            source: Kamera indeksi (0, 1, ...) veya RTSP URL / video dosya yolu.
            width: Frame genişliği.
            height: Frame yüksekliği.
            fps: Hedef FPS.
            buffer_size: Frame buffer boyutu.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        self._queue: Queue = Queue(maxsize=buffer_size)
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None

        # FPS hesaplama
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

    def start(self) -> "CameraStream":
        """Asenkron frame okumayı başlatır."""
        if self._running:
            return self
        self._running = True
        self._thread = Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def _read_loop(self):
        """Arka planda frame okuyan döngü."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    # Video dosyası bitti, başa sar
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Buffer doluysa eski frame'i at
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put(frame)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Son frame'i okur.

        Returns:
            (success, frame) tuple'ı.
        """
        if not self._running:
            return self.cap.read()

        if self._queue.empty():
            return False, None

        frame = self._queue.get()

        # FPS hesapla
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

        return True, frame

    @property
    def fps(self) -> float:
        """Anlık FPS değeri."""
        return self._fps

    def stop(self):
        """Stream'i durdurur."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.cap.release()

    def __del__(self):
        self.stop()
