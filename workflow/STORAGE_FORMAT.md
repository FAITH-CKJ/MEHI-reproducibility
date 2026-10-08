LOSSLESS NUMERICAL TABLE FORMAT
===============================

Purpose
-------
The compact tables preserve the source table's row order, column order, integer
and Boolean dtypes, strings and IEEE-754 floating-point bits. The encoding does
not cast float64 values to float32, round decimal values, quantize coordinates,
or discard small residuals.

Canonical CSV input is read with:
    pandas.read_csv(path, float_precision="round_trip")

The source CSV's textual formatting and gzip wrapper are not stored. The
reconstructed DataFrame is the exact numerical table. Use the DataFrame API
when exact dtypes, signed zero or NaN payload bits are required.

Dependencies
------------
Python, NumPy, pandas and zstandard. The codec and route decoder do not require
GIS software, PyArrow, Blosc, a local project directory, or network access.

Read a complete compact table
----------------------------
Place lossless_frame_codec.py on the Python import path:

    from lossless_frame_codec import read_frame
    annual = read_frame("classification_sensitivity_inputs_150m.ndcol.zst")

The function validates compressed blocks, decoded blocks and every reconstructed
column using SHA-256. Its default verification should remain enabled.

Optional CSV export:
    python lossless_frame_codec.py decode input.ndcol.zst restored.csv

The export uses 17 significant digits. Re-read exported numerical CSVs with
float_precision="round_trip". CSV is an interchange representation; the binary
table and its schema preserve the exact original dtypes and special float bits.

Compare a compact table with its original CSV:
    python lossless_frame_codec.py verify input.ndcol.zst source.csv.gz

Generic encoding API:
    from lossless_frame_codec import write_frame
    write_frame(dataframe, "output.ndcol.zst")

The DataFrame must have unique string column names and a default RangeIndex.
Store any other index as an explicit column. Supported columns are numeric,
Boolean, and non-missing string columns.

File layout, version 1
----------------------
All multibyte values use little-endian representation.

  8 bytes: ASCII magic NDCOL001
  8 bytes: unsigned length of the compressed manifest
  that many bytes: zstd-compressed UTF-8 JSON manifest
  remaining bytes: independent zstd blocks, in manifest order

Each block records its offset relative to the block area, compressed and
uncompressed sizes, numeric storage dtype, element count, reversible transform,
and checksums. The manifest records original columns, dtypes, order, decoded
column hashes, string dictionaries and any arithmetic recipes. No Python code
is evaluated from the manifest.

The transform is selected from:
  * original byte order;
  * byte shuffle: transpose the element-by-byte matrix;
  * bit shuffle: transpose bit planes and pack them in little bit order;
  * previous-value XOR for floating-point bit patterns;
  * previous-value modular differences for integer bit patterns;
  * sparse mode: a packed position mask, the common bit pattern, and the
    original bit patterns of non-mode values.

Bit-shuffled planes are padded only at their end; element counts remove padding
during restoration. Integer differences and their inverse operate modulo
2^(8 * storage itemsize). Integer storage may be safely narrowed after checking
the full value range; original dtypes are restored. Floating-point storage is
never narrowed.

String columns use dictionaries, or a reversible two-integer representation
for strings exactly matching their reconstructed left_right form. A string
with leading zeros that would be changed by this representation uses a
dictionary instead. String hashes use each UTF-8 value preceded by its unsigned
64-bit byte length; numeric hashes use canonical little-endian array bytes.

Exact arithmetic recipes
------------------------
An optional recipe contains column references, numeric constants, and explicit
add/subtract/multiply/divide/greater/where operations. A grouped one-based row
counter is also supported. Operations are evaluated separately and dependencies
are resolved in order.

For each predicted floating-point column, encoding compares every original
IEEE-754 bit pattern with the computed prediction. If they differ, it stores:

  residual = original_bits - predicted_bits, modulo 2^64

Restoration performs the inverse modular integer addition and views the result
as float64. The stored residual is an integer bit-pattern correction, not a
floating-point subtraction, rounded error, tolerance, or scientific adjustment.
The final column hash verifies the original bits. A zero residual can be omitted
only when every row was exactly equal at encoding.

In the annual component table, the duplicate mangrove-area column, external
area sum, P/B/S components, MEHI and yearly GRID_UID sequence can all be rebuilt
with zero residual under their stored ordered formulas. Remaining numeric
columns retain their original bits in the compressed blocks.

Route table relative to annual data
----------------------------------
The complete 25-column route table has a separate decoder:

    import sys
sys.path.insert(0, "code")
from restore_route_table import read_route
    route = read_route(
        annual,
        "data/packed/route_relative_payload.ndcol.zst",
        "metadata/route_relative_recipe.json",
    )

The payload stores integer row links to the 2000 and 2023 annual records,
essential transition measurements and categorical fields, and exact bit
residuals. The decoder checks endpoint years and matching GU_A3/GRID_ID keys,
then verifies every restored source column against its hash.

The formulas use:
  * pMEHI: half the annual MEHI difference;
  * weight: minimum endpoint mangrove area;
  * delta_P/C/B/S: the corresponding annual component differences;
  * class-share change: class_area_2023 / total_observed_area_2023 minus
    class_area_2000 / total_observed_area_2000;
  * soft-share change: the ordered sum of its three class-share changes;
  * replacement shares: stored soft-to-built area divided by the stated
    transition-area denominator;
  * evidence: the specified route's component/ratio or maximum class loss.

Every nonzero bit residual is retained. Total observed area means A_M_M2 plus
A_OBS_EXT_M2; the class-share formulas do not substitute a constant 22500 m2
denominator. Transition overlay area has its own stored corrections to a
minimum-endpoint-area predictor. These corrections preserve genuine geometric
differences as well as rounding differences; they are not treated as numerical
noise or discarded. The route recipe JSON contains only schema, formula
definitions, years and column hashes.

The signed-zero differences in route evidence are also retained, even when
their numerical difference is zero.

Integrity and independence
--------------------------
A restored table is accepted only after shape, column order, original dtypes,
strings and every numeric bit pattern agree with the source. Each compact file
can be read without the original CSV; route restoration additionally requires
the linked annual DataFrame and the small route recipe JSON.
