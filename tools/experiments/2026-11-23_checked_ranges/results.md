## Vague events: mean extra, share of cells harmed, and the difference from not told with its interval

| model | reasoning | twin | arm | runs | extra | not told | harmed | difference | 95 % interval |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:120b | low | incubator | number | 5 | 4.65 | 8.50 | 6 % | -3.85 | -7.99 to +0.41 |
| gpt-oss:120b | low | incubator | range | 5 | 2.37 | 8.50 | 6 % | -6.13 | -9.13 to -3.29 |
| gpt-oss:120b | low | incubator | checked | 5 | 1.58 | 8.50 | 0 % | -6.92 | -9.89 to -3.79 |
| gpt-oss:120b | low | bikeshare | number | 5 | 2.98 | 3.20 | 18 % | -0.22 | -3.07 to +2.46 |
| gpt-oss:120b | low | bikeshare | range | 5 | 4.95 | 3.20 | 26 % | +1.76 | -0.24 to +4.68 |
| gpt-oss:120b | low | bikeshare | checked | 5 | 2.35 | 3.20 | 12 % | -0.84 | -3.39 to +1.11 |
| gpt-oss:120b | medium | incubator | number | 5 | 6.55 | 8.50 | 9 % | -1.95 | -6.72 to +3.39 |
| gpt-oss:120b | medium | incubator | range | 5 | 3.27 | 8.50 | 9 % | -5.22 | -8.80 to -1.67 |
| gpt-oss:120b | medium | incubator | checked | 5 | 1.82 | 8.50 | 3 % | -6.68 | -9.64 to -3.72 |
| gpt-oss:120b | medium | bikeshare | number | 5 | 2.48 | 3.20 | 17 % | -0.71 | -3.33 to +1.42 |
| gpt-oss:120b | medium | bikeshare | range | 5 | 2.42 | 3.20 | 20 % | -0.78 | -4.11 to +2.55 |
| gpt-oss:120b | medium | bikeshare | checked | 5 | 1.71 | 3.20 | 2 % | -1.49 | -3.41 to -0.33 |
| gpt-oss:20b | low | incubator | number | 5 | 10.33 | 8.50 | 12 % | +1.83 | -5.71 to +14.13 |
| gpt-oss:20b | low | incubator | range | 5 | 3.57 | 8.50 | 3 % | -4.93 | -7.76 to -2.33 |
| gpt-oss:20b | low | incubator | checked | 5 | 4.41 | 8.50 | 9 % | -4.09 | -7.89 to -0.40 |
| gpt-oss:20b | low | bikeshare | number | 5 | 1.36 | 3.20 | 3 % | -1.83 | -4.04 to -0.39 |
| gpt-oss:20b | low | bikeshare | range | 5 | 2.84 | 3.20 | 12 % | -0.35 | -2.80 to +1.74 |
| gpt-oss:20b | low | bikeshare | checked | 5 | 1.51 | 3.20 | 5 % | -1.69 | -3.57 to -0.40 |
| qwen3.5:9b | none | incubator | number | 5 | 11.94 | 8.50 | 16 % | +3.44 | -5.08 to +15.79 |
| qwen3.5:9b | none | incubator | range | 5 | 4.15 | 8.50 | 6 % | -4.35 | -7.99 to -0.77 |
| qwen3.5:9b | none | incubator | checked | 5 | 5.38 | 8.50 | 0 % | -3.12 | -4.69 to -1.52 |
| qwen3.5:9b | none | bikeshare | number | 5 | 3.64 | 3.20 | 14 % | +0.45 | -3.14 to +4.80 |
| qwen3.5:9b | none | bikeshare | range | 5 | 7.19 | 3.20 | 23 % | +3.99 | +0.00 to +8.41 |
| qwen3.5:9b | none | bikeshare | checked | 5 | 5.99 | 3.20 | 18 % | +2.79 | -0.12 to +7.05 |

## The registered questions (by the means)

| model | reasoning | twin | 1. checked below not told | 2. checked below range | 3. checked harms fewer cells than range |
|---|---|---|---|---|---|
| gpt-oss:120b | low | incubator | yes (1.58 / 8.50) | yes (1.58 / 2.37) | yes (0 % / 6 %) |
| gpt-oss:120b | low | bikeshare | yes (2.35 / 3.20) | yes (2.35 / 4.95) | yes (12 % / 26 %) |
| gpt-oss:120b | medium | incubator | yes (1.82 / 8.50) | yes (1.82 / 3.27) | yes (3 % / 9 %) |
| gpt-oss:120b | medium | bikeshare | yes (1.71 / 3.20) | yes (1.71 / 2.42) | yes (2 % / 20 %) |
| gpt-oss:20b | low | incubator | yes (4.41 / 8.50) | no (4.41 / 3.57) | no (9 % / 3 %) |
| gpt-oss:20b | low | bikeshare | yes (1.51 / 3.20) | yes (1.51 / 2.84) | yes (5 % / 12 %) |
| qwen3.5:9b | none | incubator | yes (5.38 / 8.50) | no (5.38 / 4.15) | yes (0 % / 6 %) |
| qwen3.5:9b | none | bikeshare | no (5.99 / 3.20) | yes (5.99 / 7.19) | yes (18 % / 23 %) |

## Bike share by kind of event: mean extra (share harmed)

| model | reasoning | arm | 21-24, one vague input | 25-28, the same situations, all vague | 29-31, whole day | not told: 21-28, 29-31 |
|---|---|---|---|---|---|---|
| gpt-oss:120b | low | number | 0.44 (0 %) | 6.87 (33 %) | 1.16 (22 %) | 4.30, 0.24 |
| gpt-oss:120b | low | range | 3.06 (0 %) | 9.42 (46 %) | 1.52 (33 %) | 4.30, 0.24 |
| gpt-oss:120b | low | checked | 0.44 (0 %) | 5.06 (21 %) | 1.30 (17 %) | 4.30, 0.24 |
| gpt-oss:120b | medium | number | 0.44 (0 %) | 6.01 (33 %) | 0.50 (17 %) | 4.30, 0.24 |
| gpt-oss:120b | medium | range | 0.16 (0 %) | 5.93 (33 %) | 0.74 (28 %) | 4.30, 0.24 |
| gpt-oss:120b | medium | checked | 0.74 (0 %) | 3.83 (4 %) | 0.17 (0 %) | 4.30, 0.24 |
| gpt-oss:20b | low | number | 0.44 (0 %) | 3.08 (4 %) | 0.31 (6 %) | 4.30, 0.24 |
| gpt-oss:20b | low | range | 1.24 (4 %) | 5.85 (21 %) | 0.97 (11 %) | 4.30, 0.24 |
| gpt-oss:20b | low | checked | 2.47 (0 %) | 1.53 (12 %) | 0.19 (0 %) | 4.30, 0.24 |
| qwen3.5:9b | none | number | 0.68 (0 %) | 9.16 (38 %) | 0.24 (0 %) | 4.30, 0.24 |
| qwen3.5:9b | none | range | 4.30 (0 %) | 15.28 (62 %) | 0.24 (0 %) | 4.30, 0.24 |
| qwen3.5:9b | none | checked | 4.20 (0 %) | 12.08 (50 %) | 0.24 (0 %) | 4.30, 0.24 |

## Answers

1. `checked` beats not telling the twin: **no** (7 of 8 comparisons).
2. `checked` beats `range`: **no** (6 of 8 comparisons).
3. `checked` harms fewer cells than `range`: **no** (7 of 8 comparisons).
4. one vague input is easier than several (21-24 below 25-28, for `range` and `checked`): **no** (7 of 8 comparisons).

## Reading the other events (right, of the cells; mean over runs)

| model | reasoning | arm | stated | nothing | inexpressible | vague cells without an action |
|---|---|---|---|---|---|---|
| gpt-oss:120b | low | number | 10.0 / 10 | 10.0 / 10 | 7.0 / 10 | 7.0 / 98 |
| gpt-oss:120b | low | range | 10.0 / 10 | 10.0 / 10 | 7.0 / 10 | 14.0 / 98 |
| gpt-oss:120b | low | checked | 10.0 / 10 | 10.0 / 10 | 6.0 / 10 | 9.0 / 98 |
| gpt-oss:120b | medium | number | 10.0 / 10 | 10.0 / 10 | 6.0 / 10 | 3.0 / 98 |
| gpt-oss:120b | medium | range | 10.0 / 10 | 10.0 / 10 | 10.0 / 10 | 10.0 / 98 |
| gpt-oss:120b | medium | checked | 10.0 / 10 | 10.0 / 10 | 10.0 / 10 | 14.0 / 98 |
| gpt-oss:20b | low | number | 10.0 / 10 | 10.0 / 10 | 9.0 / 10 | 18.0 / 98 |
| gpt-oss:20b | low | range | 10.0 / 10 | 10.0 / 10 | 10.0 / 10 | 17.0 / 98 |
| gpt-oss:20b | low | checked | 10.0 / 10 | 10.0 / 10 | 9.0 / 10 | 13.0 / 98 |
| qwen3.5:9b | none | number | 8.0 / 10 | 10.0 / 10 | 10.0 / 10 | 32.0 / 98 |
| qwen3.5:9b | none | range | 0.0 / 10 | 10.0 / 10 | 3.0 / 10 | 36.0 / 98 |
| qwen3.5:9b | none | checked | 6.0 / 10 | 10.0 / 10 | 9.0 / 10 | 24.6 / 98 |
