from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "datasets" / "raw"
DEFAULT_ARTIFACTS_ROOT = ROOT / "artifacts"
DEFAULT_TRAINING_OUTPUT_DIR = DEFAULT_ARTIFACTS_ROOT / "training"
DEFAULT_BENCHMARK_OUTPUT_DIR = DEFAULT_ARTIFACTS_ROOT / "benchmark"


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_for_root(data_root: str | Path = DEFAULT_DATA_ROOT, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    data_root = Path(data_root).expanduser().resolve()
    candidates = (
        data_root / "the-fake-or-real-dataset",
        data_root / "the-fake-or-real-dataset" / "the-fake-or-real-dataset",
    )
    return _first_existing(*candidates) or candidates[0]


def resolve_in_the_wild_root(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    data_root = Path(data_root).expanduser().resolve()
    candidates = (
        data_root / "release-in-the-wild" / "release_in_the_wild",
        data_root / "release-in-the-wild",
        data_root / "in-the-wild-audio-deepfake" / "release_in_the_wild",
        data_root / "in-the-wild-audio-deepfake",
    )
    return _first_existing(*candidates) or candidates[0]


def resolve_asvspoof_root(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    data_root = Path(data_root).expanduser().resolve()
    candidates = (
        data_root / "asvpoof-2019-dataset" / "LA",
        data_root / "asvpoof-2019-dataset" / "ASVspoof2019_root" / "LA",
        data_root / "asvspoof-2019" / "LA",
    )
    return _first_existing(*candidates) or candidates[0]


def resolve_wavefake_root(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    data_root = Path(data_root).expanduser().resolve()
    candidates = (
        data_root / "wavefake-test" / "generated_audio",
        data_root / "wavefake-dataset" / "generated_audio",
        data_root / "wavefake-test",
    )
    return _first_existing(*candidates) or candidates[0]


def resolve_librisvoc_root(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    data_root = Path(data_root).expanduser().resolve()
    candidates = (
        data_root / "librisvoc" / "test-clean",
        data_root / "LibriSVoC" / "test-clean",
        data_root / "librisvoc",
    )
    return _first_existing(*candidates) or candidates[0]


def resolve_training_checkpoint_path(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()

    candidates = (
        DEFAULT_TRAINING_OUTPUT_DIR / "best_model.pth",
        ROOT / "best_model.pth",
    )
    return _first_existing(*candidates) or candidates[0]
