"""Model artifacts on disk.

MLflow records *experiments*; this module records the one model the running
system actually uses. Keeping them separate means the API and worker never
need an MLflow server at runtime -- they read a file and a small JSON
manifest.

Layout:

    models/
      xgb-all166/
        model.ubj          XGBoost native format
        metadata.json      version, split, threshold, metrics, mlflow run id
      graphsage/
        model.pt           state dict
        metadata.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def _default_models_dir() -> Path:
    """Find the models directory by walking up from this file.

    A fixed relative depth does not work: on the host the package sits at
    `<repo>/backend/argus/ml/`, but in the container it is `/app/argus/ml/`
    with the models bind-mounted at `/app/models`. Counting parents finds
    `<repo>/models` in one and `/models` in the other. Walking up until a
    `models` directory exists gets both right.
    """
    override = os.environ.get("ARGUS_MODELS_DIR")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    candidates = [parent / "models" for parent in here.parents]

    # Prefer a directory that actually holds a model, so a stray empty
    # `models/` closer to the package cannot shadow the real store.
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/metadata.json")):
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return here.parents[2].parent / "models"


MODELS_DIR = _default_models_dir()


@dataclass
class ModelMetadata:
    """Everything needed to trace a score back to how it was produced."""

    name: str
    model_type: str  # xgboost | graphsage
    feature_variant: str
    version: str
    n_features: int
    splits: dict[str, str]
    # The decision threshold chosen on validation at the alert budget. Null
    # for GraphSAGE, which does not gate the queue.
    threshold: float | None
    alert_budget: float
    hyperparameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    scaler: dict | None = None
    mlflow_run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def model_dir(name: str, root: Path | None = None) -> Path:
    return (root or MODELS_DIR) / name


def save_metadata(metadata: ModelMetadata, root: Path | None = None) -> Path:
    directory = model_dir(metadata.name, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return path


def load_metadata(name: str, root: Path | None = None) -> ModelMetadata:
    path = model_dir(name, root) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"no model metadata at {path}")
    return ModelMetadata(**json.loads(path.read_text(encoding="utf-8")))


def save_xgboost(model, metadata: ModelMetadata, root: Path | None = None) -> Path:
    directory = model_dir(metadata.name, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.ubj"
    model.save_model(str(path))
    save_metadata(metadata, root)
    return path


def save_torch(state_dict, metadata: ModelMetadata, root: Path | None = None) -> Path:
    import torch

    directory = model_dir(metadata.name, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.pt"
    torch.save(state_dict, path)
    save_metadata(metadata, root)
    return path
