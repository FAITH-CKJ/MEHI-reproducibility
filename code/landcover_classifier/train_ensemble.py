"""Evaluate the retained seven-checkpoint decision rule on a labelled table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .dataset import LandCoverTableDataset
    from .inference_ensemble import load_ensemble, predict_tensor, resolve_device
except ImportError:
    from dataset import LandCoverTableDataset
    from inference_ensemble import load_ensemble, predict_tensor, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset = LandCoverTableDataset(args.validation_csv)
    device = resolve_device(args.device)
    models = load_ensemble(args.models_dir, device)
    predictions = predict_tensor(dataset.features, models, device) - 1
    labels = dataset.labels
    matrix = torch.zeros((6, 6), dtype=torch.long)
    for truth, predicted in zip(labels, predictions):
        matrix[int(truth), int(predicted)] += 1
    total = float(matrix.sum())
    observed = float(matrix.diag().sum() / total)
    expected = float((matrix.sum(0) * matrix.sum(1)).sum() / (total * total))
    precision = []
    recall = []
    for class_id in range(6):
        col = int(matrix[:, class_id].sum())
        row = int(matrix[class_id, :].sum())
        precision.append(float(matrix[class_id, class_id] / col) if col else 0.0)
        recall.append(float(matrix[class_id, class_id] / row) if row else 0.0)
    result = {
        "n_validation": int(total),
        "overall_accuracy": observed,
        "cohen_kappa": (observed - expected) / (1.0 - expected),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix.tolist(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
