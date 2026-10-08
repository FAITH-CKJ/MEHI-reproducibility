Current land-cover validation record for CEE R1

The active training/validation accounting is 42,795 + 10,657 = 53,452 samples.
Map evaluation counts every coordinate-year once. There are 192 duplicate
validation records (all labels agree) and 654 coordinates outside the maps.
The final accuracy matrix uses 9,811 unique covered coordinate-year points.
These are explicit filters of the one retained validation table; sample rows
and unique evaluated points have distinct, recorded roles. No boundary ambiguity
or cross-split coordinate-year overlap occurs. Current accuracy and kappa are
recorded in validation_record.json and recomputed directly from this matrix.
The annual pooled and full year-specific matrices use rows=mapped class and
columns=reference label. Class-specific user's accuracy divides by row totals;
producer's accuracy divides by reference-column totals.

The original generated mosaics and production training CSVs remain unchanged.
This record supersedes the historical validation tables in the frozen review
package for all revised-paper accuracy and composition-sensitivity statements.
The evaluated validation set is disjoint from training coordinates within years.
It also served the retained model-selection workflow; no independent third test
set, spatial block test or leave-year-out evaluation is claimed.
