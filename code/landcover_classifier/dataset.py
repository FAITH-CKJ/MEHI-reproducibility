"""Tabular input loader for the 17-predictor land-cover classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


PREDICTOR_COLUMNS = [
    "red",
    "green",
    "blue",
    "nir",
    "swir1",
    "swir2",
    "lst",
    "ndvi",
    "evi",
    "ndbi",
    "ndpi",
    "bsi",
    "ndwi",
    "mndwi",
    "cmri",
    "mmri",
    "dtw",
]

CLASS_NAMES = {
    1: "mangrove",
    2: "tidal_flat",
    3: "non_mangrove_vegetation",
    4: "water",
    5: "built_up",
    6: "other",
}


def read_predictor_table(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = PREDICTOR_COLUMNS + (["Landcover"] if require_labels else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = np.isfinite(frame[PREDICTOR_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    if require_labels:
        valid &= frame["Landcover"].isin(CLASS_NAMES)
    return frame.loc[valid].reset_index(drop=True)


class LandCoverTableDataset(TorchDataset):
    def __init__(self, path: str | Path):
        self.frame = read_predictor_table(path, require_labels=True)
        self.features = torch.as_tensor(
            self.frame[PREDICTOR_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.labels = torch.as_tensor(
            self.frame["Landcover"].to_numpy(dtype=np.int64) - 1, dtype=torch.long
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"feat": self.features[index], "label": self.labels[index]}

    def __len__(self) -> int:
        return len(self.frame)
