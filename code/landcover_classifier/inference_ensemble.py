"""Exact seven-checkpoint inference rule used for the six-class mosaics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd

try:
    from .dataset import CLASS_NAMES, PREDICTOR_COLUMNS, read_predictor_table
    from .model import Model, load_checkpoint
except ImportError:
    from dataset import CLASS_NAMES, PREDICTOR_COLUMNS, read_predictor_table
    from model import Model, load_checkpoint


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    feature_indices: tuple[int, ...]
    selection_target: int | None


ALL_FEATURES = tuple(range(17))
MODEL_SPECS = (
    ModelSpec("specialist_1", "0_new_model_parameters.pth", tuple(range(9)) + (14, 15, 16), 0),
    ModelSpec("specialist_2", "1_new_model_parameters.pth", tuple(range(7)) + (11, 16), 1),
    ModelSpec("specialist_3", "2_new_model_parameters.pth", tuple(range(9)) + (16,), 2),
    ModelSpec("specialist_4", "3_new_model_parameters.pth", tuple(range(7)) + (12, 13), 3),
    ModelSpec("specialist_5", "4_new_model_parameters.pth", tuple(range(7)) + (9, 11), 4),
    ModelSpec("specialist_6", "5_model_parameters.pth", ALL_FEATURES, 5),
    ModelSpec("all_class", "6_model_parameters.pth", ALL_FEATURES, None),
)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_ensemble(models_dir: str | Path, device: torch.device) -> list[Model]:
    models_dir = Path(models_dir)
    models = []
    for spec in MODEL_SPECS:
        checkpoint = models_dir / spec.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = load_checkpoint(checkpoint, device)
        if model.input_dim != len(spec.feature_indices):
            raise ValueError(
                f"{spec.checkpoint}: checkpoint input dimension {model.input_dim} "
                f"does not match the documented subset of {len(spec.feature_indices)} predictors"
            )
        models.append(model)
    return models


def predict_tensor(
    features: torch.Tensor,
    models: list[Model],
    device: torch.device,
    batch_size: int = 65536,
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 17:
        raise ValueError("features must have shape (n, 17)")
    output_batches = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            predictions = [
                model(batch[:, spec.feature_indices]).argmax(dim=1)
                for model, spec in zip(models, MODEL_SPECS)
            ]
        final = predictions[6].clone()
        final = torch.where(predictions[1] == 1, torch.ones_like(final), final)
        final = torch.where(predictions[0] == 0, torch.zeros_like(final), final)
        final = torch.where(predictions[3] == 3, torch.full_like(final, 3), final)
        final = torch.where(predictions[5] == 5, torch.full_like(final, 5), final)
        output_batches.append(final.cpu() + 1)
    return torch.cat(output_batches) if output_batches else torch.empty(0, dtype=torch.long)


def predict_numpy(
    features: np.ndarray,
    models: list[Model],
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    tensor = torch.as_tensor(features, dtype=torch.float32)
    return predict_tensor(tensor, models, device, batch_size).numpy().astype(np.uint8)


def self_test(models_dir: Path, device_name: str) -> None:
    device = resolve_device(device_name)
    models = load_ensemble(models_dir, device)
    torch.manual_seed(20260720)
    test_features = torch.randn(32, 17)
    predictions = predict_tensor(test_features, models, device, batch_size=7)
    assert predictions.shape == (32,)
    assert int(predictions.min()) >= 1 and int(predictions.max()) <= 6
    print(f"PASS: loaded seven checkpoints on {device}; predicted 32 rows in classes 1--6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.models_dir, args.device)
        return
    if args.input_csv is None or args.output_csv is None:
        parser.error("--input-csv and --output-csv are required unless --self-test is used")

    original = pd.read_csv(args.input_csv)
    valid = read_predictor_table(args.input_csv, require_labels=False)
    device = resolve_device(args.device)
    models = load_ensemble(args.models_dir, device)
    predicted = predict_numpy(
        valid[PREDICTOR_COLUMNS].to_numpy(dtype=np.float32), models, device, args.batch_size
    )
    valid["predicted_class_code"] = predicted
    valid["predicted_class"] = [CLASS_NAMES[int(code)] for code in predicted]
    key_columns = [column for column in original.columns if column not in PREDICTOR_COLUMNS]
    output = valid[key_columns + ["predicted_class_code", "predicted_class"]]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
