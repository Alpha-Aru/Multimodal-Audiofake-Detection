# Dataset Setup

This repository's research scripts reference several large audio datasets. The confirmed Kaggle sources are listed in [kaggle_datasets.json](/Users/aryamansarda/Desktop/Programming/Multimodal-Audiofake-Detection/datasets/kaggle_datasets.json).

## Confirmed Kaggle Datasets

- FoR: `mohammedabdeldayem/the-fake-or-real-dataset`
- In The Wild: `bhaveshkumars/release-in-the-wild`
- ASVspoof 2019: `awsaf49/asvpoof-2019-dataset`
- WaveFake: `andreadiubaldo/wavefake-test`

## Not Yet Mapped Cleanly

- `LibriSVoC`
  The current benchmark script references `/kaggle/input/librisvoc/test-clean`, but I did not find a clear public Kaggle dataset that matches that path. Treat this one as a manual dataset source until you confirm the exact origin.

## Downloading Locally

1. Install the Kaggle client:

```bash
pip install kaggle
```

2. Put your Kaggle API token at `~/.kaggle/kaggle.json`.

3. Download all confirmed datasets into `datasets/raw/`:

```bash
python scripts/download_kaggle_datasets.py
```

4. Or download one dataset:

```bash
python scripts/download_kaggle_datasets.py --only FoR
```

The downloader unzips each dataset into the local directory declared in the manifest.
