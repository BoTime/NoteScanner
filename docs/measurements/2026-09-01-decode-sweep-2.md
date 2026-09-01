# decode sweep — 2026-09-01T01:07:27.128Z

- adapter: apple / metal-3
- grid: `{"decodePaths":["none"],"batchSizes":[8],"dtypes":["fp32"],"pointsPerSide":[16,32],"lowResFilterNms":[false,true],"keepRawMasks":false,"reps":1}`
- rows: 4 (4 ok, 0 failed)

> `overlapDecodeFilter` does not make decode faster. It keeps the next batch on the GPU while the current one is filtered on the CPU, so time MOVES BETWEEN the stage counters. A near-zero `decode` on an overlap row means the GPU finished during the previous filter block — the work happened, it stopped being counted. Rank on `budget` (wall clock minus `model-load`). The `decode` column is not the ranking.

**Fastest: none · fp32 · batch 8 · pps 16 · lowres — 7113.0 ms budget (row `p16-fp32-b8-none-lowres-r1`).**

| rank | row | decode path | dtype | batch | pps | lowres | budget | total | model-load | encode | decode | filter | nms | resample | mask-encode | raw | afterFilter | afterNms | returned | status |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | p16-fp32-b8-none-lowres-r1 | none | fp32 | 8 | 16 | lowres | 7113.0 | 7779.2 | 666.2 | 1167.8 | 5621.7 | 176.8 | 37.5 | 45.9 | 35.7 | 768 | 240 | 34 | 34 | ok |
| 2 | p16-fp32-b8-none-fullres-r1 | none | fp32 | 8 | 16 | fullres | 7798.0 | 8448.2 | 650.2 | 1219.3 | 5531.4 | 693.0 | 292.0 | 0.0 | 32.4 | 768 | 242 | 36 | 36 | ok |
| 3 | p32-fp32-b8-none-lowres-r1 | none | fp32 | 8 | 32 | lowres | 24542.6 | 25171.8 | 629.2 | 1267.4 | 22363.7 | 658.5 | 103.4 | 68.5 | 50.5 | 3072 | 947 | 53 | 53 | ok |
| 4 | p32-fp32-b8-none-fullres-r1 | none | fp32 | 8 | 32 | fullres | 27020.2 | 27671.3 | 651.1 | 1164.2 | 22084.4 | 2647.0 | 1045.7 | 0.0 | 50.3 | 3072 | 951 | 57 | 57 | ok |

All times in ms. `budget` = `total` − `model-load`; every row respawns the worker and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over 7 phases. `raw`/`afterFilter`/`afterNms`/`returned` are mask counts: a row that is fast because it silently dropped masks is visible here, and a gap between `afterNms` and `returned` is the full-resolution area re-check.
