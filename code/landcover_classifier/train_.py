"""Train and select the seven retained MLP checkpoints from labelled CSV tables."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import LandCoverTableDataset
    from .inference_ensemble import MODEL_SPECS, resolve_device
    from .model import Model
except ImportError:
    from dataset import LandCoverTableDataset
    from inference_ensemble import MODEL_SPECS, resolve_device
    from model import Model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((6, 6), dtype=torch.long)
    for truth, predicted in zip(labels.view(-1), predictions.view(-1)):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def score_model(model: Model, loader: DataLoader, indices: tuple[int, ...], device: torch.device):
    model.eval()
    matrix = torch.zeros((6, 6), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = model(batch["feat"].to(device)[:, indices])
            matrix += confusion_matrix(labels.cpu(), logits.argmax(dim=1).cpu())
    total = int(matrix.sum())
    overall = float(matrix.diag().sum() / total) if total else 0.0
    precision = []
    recall = []
    for class_id in range(6):
        col = int(matrix[:, class_id].sum())
        row = int(matrix[class_id, :].sum())
        precision.append(float(matrix[class_id, class_id] / col) if col else 0.0)
        recall.append(float(matrix[class_id, class_id] / row) if row else 0.0)
    return matrix, overall, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--model", choices=[str(i) for i in range(7)] + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=199)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset = LandCoverTableDataset(args.train_csv)
    validation_loader = DataLoader(
        LandCoverTableDataset(args.validation_csv), batch_size=256, shuffle=False
    )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    selected = range(7) if args.model == "all" else [int(args.model)]
    run_record = []

    for model_id in selected:
        set_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, generator=generator
        )
        spec = MODEL_SPECS[model_id]
        model = Model(len(spec.feature_indices), 6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150, 180], gamma=0.1
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.ones(6, dtype=torch.float32, device=device)
        )
        best_score = -np.inf
        best_record = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                features = batch["feat"].to(device)[:, spec.feature_indices]
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            matrix, overall, precision, recall = score_model(
                model, validation_loader, spec.feature_indices, device
            )
            selection_score = (
                overall if spec.selection_target is None else precision[spec.selection_target]
            )
            if selection_score > best_score:
                best_score = selection_score
                torch.save(model.state_dict(), args.models_dir / spec.checkpoint)
                best_record = {
                    "model": spec.name,
                    "checkpoint": spec.checkpoint,
                    "epoch": epoch,
                    "selection_score": selection_score,
                    "overall_accuracy": overall,
                    "precision": precision,
                    "recall": recall,
                    "confusion_matrix": matrix.tolist(),
                }
        run_record.append(best_record)
        print(json.dumps(best_record, ensure_ascii=True))

    record_path = args.models_dir / "training_selection_record.json"
    record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
