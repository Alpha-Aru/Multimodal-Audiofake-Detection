from __future__ import annotations

from typing import Iterable

import pywt
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .constants import (
    CHUNK_SIZE,
    FFT_HEIGHT,
    FFT_WIDTH,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WAVELET_HEIGHT,
    WAVELET_WIDTH,
)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    mean = tensor.mean()
    std = tensor.std(unbiased=False).clamp_min(1e-8)
    return (tensor - mean) / std


def load_audio_mono(audio_path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    try:
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
    except (ImportError, OSError, RuntimeError):
        audio_array, original_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)

    if original_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), original_sr, sample_rate).squeeze(0)
    return waveform


def split_audio_into_chunks(
    waveform: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> list[torch.Tensor]:
    waveform = waveform.flatten()
    if waveform.numel() == 0:
        return [torch.zeros(chunk_size, dtype=torch.float32)]

    chunks: list[torch.Tensor] = []
    for start in range(0, waveform.numel(), chunk_size):
        chunk = waveform[start : start + chunk_size]
        if chunk.numel() < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.numel()))
        chunks.append(chunk.float())

    return chunks or [torch.zeros(chunk_size, dtype=torch.float32)]


def preprocess_chunk(audio_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_tensor = audio_tensor.flatten().float()
    if audio_tensor.numel() < CHUNK_SIZE:
        audio_tensor = F.pad(audio_tensor, (0, CHUNK_SIZE - audio_tensor.numel()))
    elif audio_tensor.numel() > CHUNK_SIZE:
        audio_tensor = audio_tensor[:CHUNK_SIZE]

    x_raw = _normalize(audio_tensor)

    window = torch.hann_window(N_FFT, dtype=x_raw.dtype, device=x_raw.device)
    stft = torch.stft(
        x_raw,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        normalized=True,
        return_complex=True,
    )
    magnitude = stft.abs().pow(2).sqrt()
    magnitude = magnitude[:FFT_HEIGHT, :FFT_WIDTH]
    magnitude = F.interpolate(
        magnitude.unsqueeze(0).unsqueeze(0),
        size=(FFT_HEIGHT, FFT_WIDTH),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    x_fft = _normalize(magnitude)

    coeffs = pywt.wavedec(x_raw.cpu().numpy(), "db4", level=4)
    cA4 = torch.tensor(coeffs[0], dtype=torch.float32)
    target_wavelet_size = WAVELET_HEIGHT * WAVELET_WIDTH
    if cA4.numel() < target_wavelet_size:
        cA4 = F.pad(cA4, (0, target_wavelet_size - cA4.numel()))
    else:
        cA4 = cA4[:target_wavelet_size]
    x_wav = _normalize(cA4.view(WAVELET_HEIGHT, WAVELET_WIDTH))

    return x_raw.cpu(), x_fft.cpu(), x_wav.cpu()


def build_feature_batches(chunks: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_batch = []
    fft_batch = []
    wav_batch = []

    for chunk in chunks:
        x_raw, x_fft, x_wav = preprocess_chunk(chunk)
        raw_batch.append(x_raw)
        fft_batch.append(x_fft)
        wav_batch.append(x_wav)

    return torch.stack(raw_batch), torch.stack(fft_batch), torch.stack(wav_batch)
