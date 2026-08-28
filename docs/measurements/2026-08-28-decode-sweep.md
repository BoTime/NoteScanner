# decode sweep — 2026-08-28T04:43:12.929Z

- adapter: apple / metal-3
- grid: `{"decodePaths":["none","overlap","gpuEmbeddings","both"],"batchSizes":[8,32],"dtypes":["fp32","fp16"],"pointsPerSide":[16],"keepRawMasks":false,"reps":1}`
- rows: 16 (16 ok, 0 failed)

> `overlapDecodeFilter` does not make decode faster. It keeps the next batch on the GPU while the current one is filtered on the CPU, so time MOVES BETWEEN the stage counters. A near-zero `decode` on an overlap row means the GPU finished during the previous filter block — the work happened, it stopped being counted. Rank on `budget` (wall clock minus `model-load`). The `decode` column is not the ranking.

**Fastest: none · fp16 · batch 32 · pps 16 — 8614.6 ms budget (row `p16-fp16-b32-none-r1`).**

| rank | row | decode path | dtype | batch | pps | budget | total | model-load | encode | decode | filter | nms | mask-encode | raw | afterFilter | afterNms | status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | p16-fp16-b32-none-r1 | none | fp16 | 32 | 16 | 8614.6 | 9260.1 | 645.5 | 743.4 | 4569.4 | 2980.3 | 268.3 | 29.2 | 768 | 230 | 31 | ok |
| 2 | p16-fp16-b32-overlap-r1 | overlap | fp16 | 32 | 16 | 8685.0 | 9341.8 | 656.8 | 745.4 | 591.0 | 7029.5 | 263.7 | 29.4 | 768 | 230 | 31 | ok |
| 3 | p16-fp16-b32-both-r1 | both | fp16 | 32 | 16 | 8739.4 | 9374.0 | 634.6 | 639.6 | 686.9 | 7091.2 | 265.1 | 29.3 | 768 | 230 | 31 | ok |
| 4 | p16-fp16-b32-gpuEmbeddings-r1 | gpuEmbeddings | fp16 | 32 | 16 | 8740.3 | 9390.1 | 649.8 | 646.4 | 4623.9 | 3144.8 | 269.9 | 29.3 | 768 | 230 | 31 | ok |
| 5 | p16-fp16-b8-gpuEmbeddings-r1 | gpuEmbeddings | fp16 | 8 | 16 | 8952.2 | 9603.8 | 651.6 | 636.2 | 4634.8 | 3352.8 | 275.3 | 28.3 | 768 | 237 | 31 | ok |
| 6 | p16-fp16-b8-both-r1 | both | fp16 | 8 | 16 | 9068.9 | 9702.1 | 633.2 | 634.7 | 253.6 | 7846.4 | 281.6 | 28.5 | 768 | 237 | 31 | ok |
| 7 | p16-fp16-b8-overlap-r1 | overlap | fp16 | 8 | 16 | 9118.6 | 9761.0 | 642.4 | 742.5 | 163.7 | 7881.8 | 276.5 | 28.8 | 768 | 237 | 31 | ok |
| 8 | p16-fp32-b32-gpuEmbeddings-r1 | gpuEmbeddings | fp32 | 32 | 16 | 10063.9 | 10729.1 | 665.2 | 860.9 | 5736.0 | 3132.0 | 275.8 | 32.8 | 768 | 235 | 36 | ok |
| 9 | p16-fp32-b32-both-r1 | both | fp32 | 32 | 16 | 10111.8 | 10790.1 | 678.3 | 897.9 | 931.7 | 7950.4 | 271.5 | 32.5 | 768 | 235 | 36 | ok |
| 10 | p16-fp32-b32-overlap-r1 | overlap | fp32 | 32 | 16 | 10139.3 | 10805.2 | 665.9 | 1051.6 | 725.8 | 8035.0 | 265.6 | 32.3 | 768 | 235 | 36 | ok |
| 11 | p16-fp32-b8-both-r1 | both | fp32 | 8 | 16 | 10422.8 | 11087.3 | 664.5 | 981.5 | 372.6 | 8728.5 | 281.7 | 32.1 | 768 | 242 | 36 | ok |
| 12 | p16-fp16-b8-none-r1 | none | fp16 | 8 | 16 | 10453.7 | 13402.6 | 2948.9 | 1468.5 | 5276.9 | 3378.5 | 274.4 | 28.4 | 768 | 237 | 31 | ok |
| 13 | p16-fp32-b32-none-r1 | none | fp32 | 32 | 16 | 10470.2 | 11167.6 | 697.4 | 1202.8 | 5730.2 | 3208.8 | 267.9 | 32.4 | 768 | 235 | 36 | ok |
| 14 | p16-fp32-b8-overlap-r1 | overlap | fp32 | 8 | 16 | 10558.5 | 11194.1 | 635.6 | 1200.6 | 210.3 | 8814.7 | 275.9 | 31.6 | 768 | 242 | 36 | ok |
| 15 | p16-fp32-b8-none-r1 | none | fp32 | 8 | 16 | 10609.7 | 11230.8 | 621.1 | 1269.8 | 5572.0 | 3435.2 | 274.0 | 31.8 | 768 | 242 | 36 | ok |
| 16 | p16-fp32-b8-gpuEmbeddings-r1 | gpuEmbeddings | fp32 | 8 | 16 | 10688.7 | 11329.0 | 640.3 | 997.5 | 5720.8 | 3633.5 | 279.5 | 31.8 | 768 | 242 | 36 | ok |

All times in ms. `budget` = `total` − `model-load`; every row respawns the worker and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over 6 phases. `raw`/`afterFilter`/`afterNms` are mask counts: a row that is fast because it silently dropped masks is visible here.
