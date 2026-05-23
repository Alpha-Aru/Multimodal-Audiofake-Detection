from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download

from .constants import CLASS_LABELS, DEFAULT_MODEL_REPO, FAKE_CLASS_INDEX, SAMPLE_RATE
from .features import build_feature_batches, load_audio_mono, split_audio_into_chunks
from .modeling import TBranchDetector

CHECKPOINT_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
CHECKPOINT_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "best_model.pth",
    "best_model.pt",
    "model.pth",
    "model.pt",
)
CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")


@dataclass(slots=True)
class PredictionResult:
    label: str
    confidence: float
    fake_probability: float
    human_probability: float
    chunk_count: int
    duration_seconds: float
    suspicious_chunk_probability: float
    chunk_probabilities: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pick_checkpoint_file(repo_files: list[str]) -> str:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate in repo_files:
            return candidate

    for repo_file in repo_files:
        if repo_file.endswith(CHECKPOINT_SUFFIXES):
            return repo_file

    raise FileNotFoundError(
        "No supported checkpoint file was found in the Hugging Face repository. "
        f"Checked: {', '.join(CHECKPOINT_CANDIDATES)}"
    )


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in CHECKPOINT_KEYS:
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint did not contain a PyTorch state dict.")

    cleaned_state = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state[new_key] = value

    return cleaned_state


def _load_checkpoint_file(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return _unwrap_state_dict(load_file(checkpoint_path))

    return _unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))


class TriSpectraPredictor:
    def __init__(
        self,
        repo_id: str = DEFAULT_MODEL_REPO,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: TBranchDetector | None = None

    def _download_checkpoint(self) -> str:
        repo_files = HfApi().list_repo_files(self.repo_id, repo_type="model")
        checkpoint_name = _pick_checkpoint_file(repo_files)
        return hf_hub_download(repo_id=self.repo_id, filename=checkpoint_name, repo_type="model")

    def load_model(self) -> TBranchDetector:
        if self._model is not None:
            return self._model

        model = TBranchDetector().to(self.device)
        checkpoint_path = self._download_checkpoint()
        state_dict = _load_checkpoint_file(checkpoint_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint did not match the expected TriSpectra architecture. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )
        model.eval()
        self._model = model
        return model

    def predict_file(self, audio_path: str) -> PredictionResult:
        model = self.load_model()
        waveform = load_audio_mono(audio_path, sample_rate=SAMPLE_RATE)
        duration_seconds = waveform.numel() / SAMPLE_RATE if waveform.numel() else 0.0
        chunks = split_audio_into_chunks(waveform)
        raw_batch, fft_batch, wav_batch = build_feature_batches(chunks)

        raw_batch = raw_batch.to(self.device)
        fft_batch = fft_batch.to(self.device)
        wav_batch = wav_batch.to(self.device)

        with torch.inference_mode():
            logits = model(raw_batch, fft_batch, wav_batch)
            fake_probabilities = torch.softmax(logits, dim=1)[:, FAKE_CLASS_INDEX].detach().cpu()

        fake_probability = fake_probabilities.mean().item()
        suspicious_chunk_probability = fake_probabilities.max().item()
        label_index = FAKE_CLASS_INDEX if fake_probability >= self.threshold else 0
        confidence = fake_probability if label_index == FAKE_CLASS_INDEX else 1.0 - fake_probability

        return PredictionResult(
            label=CLASS_LABELS[label_index],
            confidence=confidence,
            fake_probability=fake_probability,
            human_probability=1.0 - fake_probability,
            chunk_count=len(chunks),
            duration_seconds=duration_seconds,
            suspicious_chunk_probability=suspicious_chunk_probability,
            chunk_probabilities=[round(value, 4) for value in fake_probabilities.tolist()],
        )

    def predict_for_gradio(self, audio_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        result = self.predict_file(audio_path)
        label_scores = {
            CLASS_LABELS[0]: result.human_probability,
            CLASS_LABELS[1]: result.fake_probability,
        }
        return label_scores, result.to_dict()
