# TriSpectra Audio Deepfake Detection

This repository packages the TriSpectra audio deepfake detector into a cleaner, shareable layout for both:

- inference through a web app or CLI
- local research workflows for caching, training, and benchmarking

The core idea is a tri-branch detector that looks at the same audio clip through three views:

- raw waveform features
- spectrogram features
- wavelet features

Those three branches are fused to classify whether a clip is likely human or a synthesized deepfake.

## Links

- Paper: <https://docs.google.com/document/d/1IriyA82F-8KW9teJCRSQbmWc7ReWa2HsL1Vw6Q2WPkM/edit?usp=sharing>
- Hugging Face model: <https://huggingface.co/skorps007/trispectra>
- Original Colab demo: <https://colab.research.google.com/drive/1Kvj7d5FSoA6zPCmxAl8YbS3kNEdqIsZT?usp=sharing>
- Kaggle FoR dataset: <https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset>
- Kaggle In The Wild dataset: <https://www.kaggle.com/datasets/bhaveshkumars/release-in-the-wild>
- Kaggle ASVspoof 2019 dataset: <https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset>
- Kaggle WaveFake dataset: <https://www.kaggle.com/datasets/andreadiubaldo/wavefake-test>

## Repository Purpose

This repo is meant to be a usable presentation layer around the research code. It contains:

- a reusable Python package under `src/trispectra`
- a Gradio web app for drag-and-drop audio uploads
- local training and benchmarking scripts under `scripts/`
- runtime loading of the published checkpoint from Hugging Face

The web app is the simplest path for someone evaluating the project. The research scripts are there if someone wants to reproduce preprocessing, retrain, or benchmark the model locally.

## How It Works

Each audio file is resampled to `16 kHz` and processed in `1-second` chunks. For every chunk, the code builds:

- a normalized raw waveform tensor
- a `128 x 128` spectrogram-style representation
- a `64 x 128` wavelet representation

The TriSpectra model passes those through separate branches and fuses them before classification. In the web app, chunk-level fake probabilities are aggregated to produce:

- a final `Human` or `Deepfake` verdict
- an overall confidence score
- the average fake probability
- the highest-risk chunk score

## Project Layout

```text
.
├── app.py                      # Web app entry point
├── pyproject.toml              # Package metadata
├── requirements.txt            # App/runtime dependencies
├── requirements-research.txt   # Local training / benchmarking dependencies
├── scripts/
│   ├── benchmark_research.py
│   ├── cache_features.py
│   ├── download_kaggle_datasets.py
│   ├── predict.py
│   └── train_research.py
├── src/
│   └── trispectra/
│       ├── __init__.py
│       ├── constants.py
│       ├── features.py
│       ├── inference.py
│       ├── modeling.py
│       └── web.py
└── tests/
```

## What The App Does

The web app accepts an uploaded audio file, splits it into 1-second chunks, runs the TriSpectra model on each chunk, and reports:

- overall verdict: `Deepfake` or `Human`
- confidence score
- average fake probability across chunks
- the most suspicious chunk score

That chunked approach is important because the training code expects 16 kHz, 1-second inputs.

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
python app.py
```

On first run, the checkpoint is downloaded from Hugging Face and cached locally.

For terminal inference against a single file:

```bash
python scripts/predict.py path/to/audio.wav
```

## Training And Benchmarking

If you want to run the research pipeline locally:

```bash
pip install -r requirements-research.txt
```

Validate the full local pathing and model forward pass without actually training:

```bash
python scripts/train_research.py --num-epochs 0
```

Train locally and write artifacts to `artifacts/training/`:

```bash
python scripts/train_research.py
```

Benchmark the trained checkpoint:

```bash
python scripts/benchmark_research.py
```

## Notes

- `scripts/` contains the original training and benchmarking workflow, now adapted to resolve local dataset folders under `datasets/raw/` by default.
- The app and inference package are the maintained path for sharing and demo use.
- CUDA is used automatically when available.

## Datasets

The research scripts depend on several external datasets. I added a manifest and downloader for the Kaggle-hosted ones:

- [datasets/kaggle_datasets.json](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/datasets/kaggle_datasets.json)
- [scripts/download_kaggle_datasets.py](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/scripts/download_kaggle_datasets.py)
- [datasets/README.md](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/datasets/README.md)

After you configure `~/.kaggle/kaggle.json`, you can fetch the confirmed Kaggle datasets with:

```bash
python scripts/download_kaggle_datasets.py
```

I confirmed Kaggle listings for:

- FoR: <https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset>
- In The Wild: <https://www.kaggle.com/datasets/bhaveshkumars/release-in-the-wild>
- ASVspoof 2019: <https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset>
- WaveFake: <https://www.kaggle.com/datasets/andreadiubaldo/wavefake-test>

`LibriSVoC` is still unresolved as a Kaggle source and remains manual for now.
