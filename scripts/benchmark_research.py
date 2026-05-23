import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import torch
import numpy as np
import librosa
import pywt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score, classification_report, roc_curve)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from trispectra.modeling import TBranchDetector
from trispectra.research_paths import (
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    resolve_asvspoof_root,
    resolve_for_root,
    resolve_librisvoc_root,
    resolve_training_checkpoint_path,
    resolve_wavefake_root,
)
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import torchaudio.transforms as T
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: torch.nn.Module | None = None
OUTPUT_DIR = DEFAULT_BENCHMARK_OUTPUT_DIR


def load_model(model_path: str | Path) -> torch.nn.Module:
    checkpoint_path = Path(model_path)
    base_model = TBranchDetector().to(device)
    base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    loaded_model: torch.nn.Module = base_model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        loaded_model = torch.nn.DataParallel(base_model)
    loaded_model.eval()
    return loaded_model

def calculate_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def preprocess(audio, sr=16000):
    audio = librosa.util.fix_length(audio, size=16000)
    x_raw = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    stft = librosa.stft(audio, n_fft=512, hop_length=256)
    mag = np.abs(stft)[:128]
    
    if mag.shape[1] > 128:
        mag = mag[:, :128]
    else:
        pad_width = ((0, 0), (0, 128 - mag.shape[1]))
        mag = np.pad(mag, pad_width, mode='constant')
    
    x_fft = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    
    # Wavelet transform
    coeffs = pywt.wavedec(audio, 'db4', level=4)
    cA4 = coeffs[0]
    cA4 = np.resize(cA4, (64*128))[:64*128]
    x_wav = (cA4.reshape(64, 128) - np.mean(cA4)) / (np.std(cA4) + 1e-8)
    
    return (
        torch.tensor(x_raw).unsqueeze(0).float(),
        torch.tensor(x_fft).unsqueeze(0).float(),
        torch.tensor(x_wav).unsqueeze(0).float()
    )

class AudioDataset(Dataset):
    def __init__(self, file_label_pairs):
        self.data = file_label_pairs
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            audio, _ = librosa.load(path, sr=16000)
            return *preprocess(audio), label
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
            return (
                torch.zeros(1, 16000), 
                torch.zeros(1, 128, 128),
                torch.zeros(1, 64, 128), 
                -1
            )

def load_for_original(base: str | Path, max_samples=None):
    base = Path(base) / "for-original" / "for-original" / "testing"
    data = []
    for label, name in [(0, 'real'), (1, 'fake')]:
        files = list((base / name).glob("*.wav"))
        if max_samples:
            files = files[:max_samples]
        data += [(str(f), label) for f in files]
    print(f"FOR-Original: Loaded {len(data)} samples")
    return data

def load_asvspoof(base_path: str | Path):
    base_path = Path(base_path)
    protocol_path = base_path / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    flac_dir = base_path / "ASVspoof2019_LA_eval" / "flac"

    print(f"Checking protocol at: {protocol_path}")
    print(f"Checking FLAC files at: {flac_dir}")
    
    file_labels = {}

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5: 
                file_id = parts[1]  
                label = 0 if parts[4] == 'bonafide' else 1
                file_labels[file_id + ".flac"] = label 

    print(f"Found {len(file_labels)} entries in protocol")

    files = []
    missing_files = 0
    for flac_file in flac_dir.glob("*.flac"):
        if flac_file.name in file_labels:
            files.append((str(flac_file), file_labels[flac_file.name]))
        else:
            missing_files += 1
    
    print(f"ASVspoof: Matched {len(files)} files (missing labels for {missing_files} files)")

    if not files:
        raise ValueError("No valid files found - check protocol/flac matching")
    return files

def load_librisvoc(base_path: str | Path):
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise ValueError(f"LibriSVoC dataset not found at: {base_path}")
    
    print(f"Found LibriSVoC at: {base_path}")
    
    files = []
    
    real_dir = base_path / "real"
    if real_dir.exists():
        real_files = list(real_dir.glob("*.wav")) + list(real_dir.glob("*.flac"))
        files.extend([(str(f), 0) for f in real_files])
        print(f"LibriSVoC: Found {len(real_files)} real files")
    
    fake_dir = base_path / "fake"
    if fake_dir.exists():
        fake_files = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.flac"))
        files.extend([(str(f), 1) for f in fake_files])
        print(f"LibriSVoC: Found {len(fake_files)} fake files")
    
    if not files:
        raise ValueError("No LibriSVoC files found. Please check real/fake folders exist.")
    
    print(f"LibriSVoC: Total loaded {len(files)} samples")
    return files

def load_wavefake(base_path: str | Path):
    base_path = Path(base_path)
    
    files = []
    
    real_dir = base_path / "ljspeech_waveglow"
    if real_dir.exists():
        real_files = list(real_dir.glob("*.wav"))
        files.extend([(str(f), 0) for f in real_files])
        print(f"WaveFake: Found {len(real_files)} real files in ljspeech_waveglow")
    
    fake_dirs = [
        "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech",
        "jsut_multi_band_melgan",
        "jsut_parallel_wavegan", 
        "ljspeech_full_band_melgan",
        "ljspeech_hifiGAN",
        "ljspeech_melgan",
        "ljspeech_melgan_large",
        "ljspeech_multi_band_melgan",
        "ljspeech_parallel_wavegan"
    ]
    
    for fake_dir_name in fake_dirs:
        fake_dir = base_path / fake_dir_name
        if fake_dir.exists():
            fake_files = list(fake_dir.glob("*.wav"))
            files.extend([(str(f), 1) for f in fake_files])
            print(f"WaveFake: Found {len(fake_files)} fake files in {fake_dir_name}")
    
    if not files:
        raise ValueError("No WaveFake files found. Please check directory structure.")
    
    print(f"WaveFake: Total loaded {len(files)} samples")
    return files

def evaluate(dataloader, name="Dataset"):
    if model is None:
        raise RuntimeError("Benchmark model has not been loaded.")
    y_true, y_prob = [], []
    
    for x_raw, x_fft, x_wav, labels in tqdm(dataloader, desc=f"Evaluating {name}"):
        x_raw = x_raw.to(device)
        x_fft = x_fft.to(device)
        x_wav = x_wav.to(device)
        labels = labels.to(device)
        
        mask = labels != -1
        if mask.sum() == 0:
            continue
            
        with torch.no_grad():
            outputs = model(x_raw[mask], x_fft[mask], x_wav[mask])
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
        y_true.extend(labels[mask].cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    
    if not y_true:
        print(f"No valid samples in {name}!")
        return None
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    eer = calculate_eer(y_true, y_prob)
    
    metrics = {
        "dataset": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "eer": float(eer),
        "classification_report": classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True),
        "num_samples": len(y_true),
        "timestamp": datetime.now().isoformat(),
        "y_true": y_true,
        "y_prob": y_prob
    }
    
    print(f"\nResults for {name}:")
    print(f"Optimal Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}\nEER = {metrics['eer']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    roc_path = OUTPUT_DIR / f"roc_{name.lower().replace('-', '_').replace(' ', '_')}.png"
    plt.savefig(roc_path)
    plt.close()
    metrics["roc_curve_path"] = str(roc_path)
    
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a local TriSpectra checkpoint on available datasets.")
    parser.add_argument("--data-root", default=str(ROOT / "datasets" / "raw"))
    parser.add_argument("--for-root", default=None)
    parser.add_argument("--asvspoof-root", default=None)
    parser.add_argument("--librisvoc-root", default=None)
    parser.add_argument("--wavefake-root", default=None)
    parser.add_argument("--model-path", default=str(resolve_training_checkpoint_path()))
    parser.add_argument("--output-dir", default=str(DEFAULT_BENCHMARK_OUTPUT_DIR))
    parser.add_argument("--max-for-original-samples", type=int, default=5000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_path)
    
    datasets = {}
    
    for_root = resolve_for_root(args.data_root, args.for_root)
    try:
        if for_root.exists():
            datasets["FOR-Original"] = load_for_original(for_root, max_samples=args.max_for_original_samples)
    except Exception as e:
        print(f"Failed to load FOR-Original: {e}")
    
    asvspoof_root = resolve_asvspoof_root(args.data_root, args.asvspoof_root)
    try:
        if asvspoof_root.exists():
            datasets["ASVspoof-2019"] = load_asvspoof(asvspoof_root)
    except Exception as e:
        print(f"Failed to load ASVspoof-2019: {e}")
    
    librisvoc_root = resolve_librisvoc_root(args.data_root, args.librisvoc_root)
    try:
        if librisvoc_root.exists():
            datasets["LibriSVoC"] = load_librisvoc(librisvoc_root)
    except Exception as e:
        print(f"Failed to load LibriSVoC: {e}")
    
    wavefake_root = resolve_wavefake_root(args.data_root, args.wavefake_root)
    try:
        if wavefake_root.exists():
            datasets["WaveFake"] = load_wavefake(wavefake_root)
    except Exception as e:
        print(f"Failed to load WaveFake: {e}")
    
    if not datasets:
        raise ValueError("No datasets were successfully loaded!")
    
    print(f"\nSuccessfully loaded {len(datasets)} datasets:")
    for name, data in datasets.items():
        print(f"- {name}: {len(data)} samples")
    
    all_metrics = {}
    
    # Evaluate each dataset
    for name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print('='*50)
        
        loader = DataLoader(
            AudioDataset(data),
            batch_size=128,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda x: (
                torch.stack([item[0] for item in x]),
                torch.stack([item[1] for item in x]),
                torch.stack([item[2] for item in x]),
                torch.tensor([item[3] for item in x])
            )
        )
        metrics = evaluate(loader, name)
        if metrics:
            all_metrics[name] = metrics
    
    metrics_path = OUTPUT_DIR / "metrics.json"
    json_metrics = {}
    for name, metrics in all_metrics.items():
        json_metrics[name] = {k: v for k, v in metrics.items() if k not in ['y_true', 'y_prob']}
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\nSaved all metrics to {metrics_path}")
    
    # Summary report
    report_path = OUTPUT_DIR / "summary.txt"
    with open(report_path, 'w') as f:
        f.write("WavSpecTR Audio Deepfake Benchmark Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Evaluation timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: WavSpecTR (Tri-Modal Architecture)\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<15} {'Samples':<8} {'Accuracy':<9} {'F1':<6} {'Precision':<9} {'Recall':<6} {'AUC':<6} {'EER':<6}\n")
        f.write("-" * 80 + "\n")
        
        for name, metrics in all_metrics.items():
            f.write(f"{name:<15} {metrics['num_samples']:<8} {metrics['accuracy']:<9.4f} "
                   f"{metrics['f1']:<6.4f} {metrics['precision']:<9.4f} {metrics['recall']:<6.4f} "
                   f"{metrics['auc']:<6.4f} {metrics['eer']:<6.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        for name, metrics in all_metrics.items():
            f.write(f"DATASET: {name}\n")
            f.write("-"*50 + "\n")
            f.write(f"Number of Samples: {metrics['num_samples']}\n")
            f.write(f"Optimal Threshold: {metrics['threshold']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['auc']:.4f}\n")
            f.write(f"Equal Error Rate (EER): {metrics['eer']:.4f}\n")
            f.write(f"ROC Curve saved to: {metrics['roc_curve_path']}\n")
            f.write("\nDetailed Classification Report:\n")
            
            y_true = metrics['y_true']
            y_pred = (np.array(metrics['y_prob']) >= metrics['threshold']).astype(int)
            report_str = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
            f.write(report_str)
            f.write("\n\n")
    
    print(f"Saved comprehensive summary report to {report_path}")
    
    # Comparison plot
    if len(all_metrics) > 1:
        plt.figure(figsize=(12, 8))
        
        datasets_names = list(all_metrics.keys())
        metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
        
        x = np.arange(len(datasets_names))
        width = 0.15
        
        for i, metric in enumerate(metrics_names):
            values = [all_metrics[dataset][metric.lower().replace('auc', 'auc')] for dataset in datasets_names]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('WavSpecTR Performance Comparison Across Datasets')
        plt.xticks(x + width*2, datasets_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        comparison_path = OUTPUT_DIR / "comparison.png"
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        print(f"Saved comparison plot to {comparison_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nBenchmarking complete!")
