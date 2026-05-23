# TriSpectra Audio Deepfake Detection

This repository contains the code for a tri-modal audio deepfake detector that combines raw waveform, spectrogram, and wavelet features. It includes:

- a local web app for running inference on uploaded audio
- the packaged inference code under `src/trispectra`
- the original training and benchmarking scripts, adjusted to run locally instead of only inside Kaggle

## References

Paper: <https://docs.google.com/document/d/1IriyA82F-8KW9teJCRSQbmWc7ReWa2HsL1Vw6Q2WPkM/edit?usp=sharing>  
Model: <https://huggingface.co/skorps007/trispectra>  
Colab demo: <https://colab.research.google.com/drive/1Kvj7d5FSoA6zPCmxAl8YbS3kNEdqIsZT?usp=sharing>

Datasets used by the research scripts:

- FoR: <https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset>
- In The Wild: <https://www.kaggle.com/datasets/bhaveshkumars/release-in-the-wild>
- ASVspoof 2019: <https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset>
- WaveFake: <https://www.kaggle.com/datasets/andreadiubaldo/wavefake-test>

`LibriSVoC` is still referenced by the benchmark script, but I did not find a clean public Kaggle source that matches the old path assumptions, so that one is still manual.

## Model Summary

The detector works on 1-second audio chunks at 16 kHz. For each chunk, the code builds three inputs:

- the normalized raw waveform
- a `128 x 128` time-frequency representation
- a `64 x 128` wavelet representation

Those are passed through separate branches and then fused for binary classification.

In the web app, longer files are split into 1-second chunks, scored independently, and then reduced to a final `Human` or `Deepfake` prediction. The UI also shows the average fake probability and the most suspicious chunk score.

## Repository Layout

```text
.
├── app.py
├── datasets/
├── pyproject.toml
├── requirements.txt
├── requirements-research.txt
├── scripts/
│   ├── benchmark_research.py
│   ├── cache_features.py
│   ├── download_kaggle_datasets.py
│   ├── predict.py
│   └── train_research.py
└── src/
    └── trispectra/
```

The main inference code lives in `src/trispectra`. The research scripts in `scripts/` are still script-like, but they now use local dataset and artifact paths by default.

## Running Inference

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

Start the web app:

```bash
python app.py
```

Run CLI inference on a single file:

```bash
python scripts/predict.py path/to/audio.wav
```

The Hugging Face checkpoint is downloaded automatically the first time it is needed.

## Training And Benchmarking

Install the research dependencies:

```bash
pip install -r requirements-research.txt
```

Download the Kaggle datasets after placing your API token at `~/.kaggle/kaggle.json`:

```bash
python scripts/download_kaggle_datasets.py
```

Check that the local training pipeline can load data and run a forward pass:

```bash
python scripts/train_research.py --num-epochs 0
```

Run training:

```bash
python scripts/train_research.py
```

Training outputs are written to `artifacts/training/`.

Run benchmarking against the trained checkpoint:

```bash
python scripts/benchmark_research.py
```

Benchmark outputs are written to `artifacts/benchmark/`.

## Dataset Notes

The local downloader and dataset manifest are here:

- [datasets/kaggle_datasets.json](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/datasets/kaggle_datasets.json)
- [datasets/README.md](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/datasets/README.md)
- [scripts/download_kaggle_datasets.py](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/scripts/download_kaggle_datasets.py)

The training script resolves datasets from `datasets/raw/` by default and skips any sources that are missing. That means you can start with only FoR if you want to verify the pipeline before pulling the larger datasets.
