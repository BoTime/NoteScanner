# decode sweep — 2026-09-01T01:05:22.476Z

- adapter: apple / metal-3
- grid: `{"decodePaths":["none"],"batchSizes":[8],"dtypes":["fp32"],"pointsPerSide":[16],"lowResFilterNms":[false,true],"keepRawMasks":true,"reps":1}`
- rows: 2 (2 ok, 0 failed)

> `overlapDecodeFilter` does not make decode faster. It keeps the next batch on the GPU while the current one is filtered on the CPU, so time MOVES BETWEEN the stage counters. A near-zero `decode` on an overlap row means the GPU finished during the previous filter block — the work happened, it stopped being counted. Rank on `budget` (wall clock minus `model-load`). The `decode` column is not the ranking.

**Fastest: none · fp32 · batch 8 · pps 16 · lowres — 8224.6 ms budget (row `p16-fp32-b8-none-lowres-r1`).**

| rank | row | decode path | dtype | batch | pps | lowres | budget | total | model-load | encode | decode | filter | nms | resample | mask-encode | raw | afterFilter | afterNms | returned | status |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | p16-fp32-b8-none-lowres-r1 | none | fp32 | 8 | 16 | lowres | 8224.6 | 8944.9 | 720.3 | 2142.4 | 5778.6 | 137.0 | 42.0 | 50.0 | 40.5 | 768 | 240 | 34 | 34 | ok |
| 2 | p16-fp32-b8-none-fullres-r1 | none | fp32 | 8 | 16 | fullres | 8275.0 | 8921.6 | 646.6 | 1390.7 | 6074.5 | 457.7 | 290.9 | 0.1 | 33.3 | 768 | 242 | 36 | 36 | ok |

All times in ms. `budget` = `total` − `model-load`; every row respawns the worker and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over 7 phases. `raw`/`afterFilter`/`afterNms`/`returned` are mask counts: a row that is fast because it silently dropped masks is visible here, and a gap between `afterNms` and `returned` is the full-resolution area re-check.

### mask agreement against the baseline row

| row | baseline masks | variant masks | matched | unmatched baseline | unmatched variant | mean IoU | median IoU | min IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p16-fp32-b8-none-fullres-r1 | 36 | 36 | 36 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| p16-fp32-b8-none-lowres-r1 | 36 | 34 | 34 | 2 | 0 | 1.000 | 1.000 | 0.993 |

Pairing is greedy best-IoU; a pair below IoU 0.9 counts as unmatched on both sides. REPORTED, NOT ENFORCED — the sweep ranks on time.
