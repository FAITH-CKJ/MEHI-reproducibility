# Six-class coastal land-cover classifier

The release contains the seven retained PyTorch checkpoints and the executable
training, table-inference, raster-inference and evaluation routines used
for the four observation-year mosaics.

The 17 predictor bands and CSV columns are ordered as follows:

`red, green, blue, nir, swir1, swir2, lst, ndvi, evi, ndbi, ndpi, bsi, ndwi, mndwi, cmri, mmri, dtw`.

Class codes are 1 mangrove, 2 tidal flat, 3 non-mangrove vegetation, 4 water,
5 built-up land and 6 other land cover. `inference_ensemble.py` records the
12-, 10-, 9- and 17-predictor subsets and the final class-priority decision
rule. `application.py` reads a co-registered 17-band raster in the order above
and writes a one-band byte classification raster. `train_.py` fits and selects
the seven model roles from separate training and held-out CSV tables, and
`train_ensemble.py` reports confusion-matrix, overall-accuracy, Kappa,
precision and recall statistics for the retained decision rule.

The supplied sample registers document sample identifiers, reference labels,
years and coordinates. They support sample accounting and final-map evaluation.
Training and application commands accept user-supplied predictor tables or
rasters with all 17 variables in the stated order. The complete annual MEHI,
matched-neighbourhood and final-map validation inputs are supplied separately
for reproducing the reported numerical results.

Run the checkpoint loading and inference self-test from the package root:

```text
python code/landcover_classifier/inference_ensemble.py --self-test
```
