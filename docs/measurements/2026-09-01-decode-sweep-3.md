# decode sweep — 2026-09-01T18:50:35.925Z

- adapter: apple / metal-3
- grid: `{"decodePaths":["none"],"batchSizes":[32],"dtypes":["fp32"],"pointsPerSide":[16,32],"lowResFilterNms":[true],"lowResMaskEncode":[false,true],"keepRawMasks":false,"reps":1}`
- rows: 4 (4 ok, 0 failed)

> `overlapDecodeFilter` does not make decode faster. It keeps the next batch on the GPU while the current one is filtered on the CPU, so time MOVES BETWEEN the stage counters. A near-zero `decode` on an overlap row means the GPU finished during the previous filter block — the work happened, it stopped being counted. Rank on `budget` (wall clock minus `model-load`). The `decode` column is not the ranking.

**Fastest: none · fp32 · batch 32 · pps 16 · lowres · enc low — 7067.2 ms budget (row `p16-fp32-b32-none-lowres-enclow-r1`).**

| rank | row | decode path | dtype | batch | pps | lowres | enc | budget | total | model-load | encode | decode | filter | nms | resample | mask-encode | raw | afterFilter | afterNms | returned | status |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | p16-fp32-b32-none-lowres-enclow-r1 | none | fp32 | 32 | 16 | lowres | enclow | 7067.2 | 7683.4 | 616.2 | 1263.6 | 5601.6 | 96.8 | 19.6 | 47.5 | 10.5 | 768 | 233 | 34 | 34 | ok |
| 2 | p16-fp32-b32-none-lowres-encfull-r1 | none | fp32 | 32 | 16 | lowres | encfull | 7092.1 | 7728.3 | 636.2 | 1318.0 | 5563.5 | 84.1 | 19.7 | 43.5 | 31.6 | 768 | 233 | 34 | 34 | ok |
| 3 | p32-fp32-b32-none-lowres-encfull-r1 | none | fp32 | 32 | 32 | lowres | encfull | 23834.4 | 24479.2 | 644.8 | 1269.9 | 22016.9 | 324.6 | 75.7 | 68.3 | 49.9 | 3072 | 916 | 53 | 53 | ok |
| 4 | p32-fp32-b32-none-lowres-enclow-r1 | none | fp32 | 32 | 32 | lowres | enclow | 23863.4 | 24505.3 | 641.9 | 1300.6 | 22018.3 | 348.8 | 77.0 | 76.4 | 14.2 | 3072 | 916 | 53 | 53 | ok |

All times in ms. `budget` = `total` − `model-load`; every row respawns the worker and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over 7 phases. `raw`/`afterFilter`/`afterNms`/`returned` are mask counts: a row that is fast because it silently dropped masks is visible here, and a gap between `afterNms` and `returned` is the full-resolution area re-check.
