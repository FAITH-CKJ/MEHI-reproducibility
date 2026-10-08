"""Apply the seven-checkpoint classifier to an aligned 17-band predictor raster."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

try:
    from .inference_ensemble import load_ensemble, predict_numpy, resolve_device
except ImportError:
    from inference_ensemble import load_ensemble, predict_numpy, resolve_device


def classify_raster(
    input_raster: Path,
    output_raster: Path,
    models_dir: Path,
    device_name: str,
    batch_size: int,
) -> None:
    device = resolve_device(device_name)
    models = load_ensemble(models_dir, device)
    with rasterio.open(input_raster) as source:
        if source.count != 17:
            raise ValueError("Input raster must contain 17 aligned bands in the documented order")
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", predictor=2)
        output_raster.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_raster, "w", **profile) as destination:
            for _, window in source.block_windows(1):
                values = source.read(window=window, out_dtype="float32")
                rows, columns = values.shape[1:]
                feature_rows = values.reshape(17, -1).T
                valid = np.isfinite(feature_rows).all(axis=1)
                for band_id, nodata in enumerate(source.nodatavals):
                    if nodata is not None and np.isfinite(nodata):
                        valid &= feature_rows[:, band_id] != nodata
                classes = np.zeros(feature_rows.shape[0], dtype=np.uint8)
                if np.any(valid):
                    classes[valid] = predict_numpy(
                        feature_rows[valid], models, device, batch_size=batch_size
                    )
                destination.write(classes.reshape(rows, columns), 1, window=window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_raster", type=Path)
    parser.add_argument("output_raster", type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).with_name("models"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    classify_raster(
        args.input_raster, args.output_raster, args.models_dir, args.device, args.batch_size
    )


if __name__ == "__main__":
    main()
