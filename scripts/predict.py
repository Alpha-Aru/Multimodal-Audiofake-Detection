from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trispectra.inference import TriSpectraPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriSpectra audio deepfake inference on a local file.")
    parser.add_argument("audio_path", help="Path to an audio file.")
    parser.add_argument("--model-repo", default="skorps007/trispectra", help="Hugging Face model repo ID.")
    args = parser.parse_args()

    predictor = TriSpectraPredictor(repo_id=args.model_repo)
    result = predictor.predict_file(args.audio_path)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
