"""HelpHand - Uluslararasi Tehlike El Isareti Tespit Sistemi.

Kullanim:
    python main.py --mode prepare              # Veriyi import et ve split'le
    python main.py --mode train                # Modeli egit (GPU)
    python main.py --mode evaluate             # Test seti uzerinde degerlendir
    python main.py --mode detect               # Gercek zamanli tespit (webcam)
    python main.py --mode detect --source 0    # Webcam
    python main.py --mode detect --source rtsp://...  # RTSP stream
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelpHand - Tehlike El Isareti Tespit Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["prepare", "train", "evaluate", "detect"],
        help="Calisma modu",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Konfigurasyon dosyasi",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Kamera kaynagi (0=webcam, rtsp://... veya video dosyasi)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model checkpoint yolu (.pth)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "prepare":
        # Veriyi D:/Verisetleri'nden al, normalize et, split'le
        from dataset.prepare import run_full_preparation
        run_full_preparation(data_dir="data")

    elif args.mode == "train":
        from training.train import train
        train(config_path=args.config)

    elif args.mode == "evaluate":
        from training.evaluate import evaluate
        evaluate(config_path=args.config, model_path=args.model)

    elif args.mode == "detect":
        import yaml
        from inference.detector import RealtimeDetector

        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        source = args.source
        if source is not None and source.isdigit():
            source = int(source)

        model_path = args.model or f"{config['training']['checkpoint_dir']}/best_model.pth"

        detector = RealtimeDetector(
            config_path=args.config,
            model_path=model_path,
        )
        detector.run(source=source)


if __name__ == "__main__":
    main()
