"""Alarm sistemi.

Tehlike işareti tespit edildiğinde alarm tetikler.
Cooldown mekanizması ile tekrarlı alarmları önler.
"""

import time
import platform
from pathlib import Path
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger("alert")


class AlertManager:
    """Alarm yönetim sistemi.

    Alarm tetikleme, cooldown, loglama ve bildirim.
    """

    def __init__(
        self,
        cooldown: float = 10.0,
        log_dir: str = "logs",
        enable_sound: bool = True,
    ):
        """
        Args:
            cooldown: Ardışık alarmlar arası minimum süre (saniye).
            log_dir: Alarm log dosyalarının dizini.
            enable_sound: Ses alarmı etkin mi.
        """
        self.cooldown = cooldown
        self.enable_sound = enable_sound
        self._last_alert_time = 0.0
        self._alert_count = 0

        # Alarm log dosyası
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._alert_log = self.log_dir / "alerts.log"

    def trigger(self, confidence: float = 0.0, details: str = "") -> bool:
        """Alarm tetikler (cooldown kontrolü ile).

        Args:
            confidence: Tespit güven skoru.
            details: Ek detay bilgisi.

        Returns:
            True ise alarm tetiklendi, False ise cooldown'da.
        """
        now = time.time()
        if now - self._last_alert_time < self.cooldown:
            return False

        self._last_alert_time = now
        self._alert_count += 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"[ALARM #{self._alert_count}] {timestamp} | "
            f"Guven: {confidence:.2f} | {details}"
        )

        # Konsol uyarısı
        logger.warning(f"\n{'!'*60}")
        logger.warning(f"  TEHLIKE ISARETI TESPIT EDILDI!")
        logger.warning(f"  {message}")
        logger.warning(f"{'!'*60}\n")

        # Dosyaya kaydet
        with open(self._alert_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

        # Ses alarmı
        if self.enable_sound:
            self._play_sound()

        return True

    def _play_sound(self):
        """Platform'a göre ses çalar."""
        try:
            if platform.system() == "Windows":
                import winsound
                # 3 kısa bip
                for _ in range(3):
                    winsound.Beep(1000, 200)
                    time.sleep(0.1)
            else:
                # Linux/Mac: terminal bip
                print("\a", end="", flush=True)
        except Exception:
            pass  # Ses çalınamazsa sessizce devam et

    @property
    def alert_count(self) -> int:
        """Toplam tetiklenen alarm sayısı."""
        return self._alert_count

    def reset(self):
        """Alarm sayacını sıfırlar."""
        self._alert_count = 0
        self._last_alert_time = 0.0
