The frozen rule, recomputed, gives the saved extra in every cell: yes

## A. Choice rules on the saved range readings (exploratory, not registered): mean extra, vague cells

| model | reasoning | twin | cells | frozen | minimax_all | mean | linked | middle | not told |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:120b | high | bikeshare | 240 | 3.09 | 2.96 | 2.74 | 2.68 | 2.77 | 2.30 |
| gpt-oss:120b | high | incubator | 160 | 9.49 | 9.50 | 9.34 | 9.49 | 9.80 | 6.00 |
| gpt-oss:120b | low | bikeshare | 240 | 4.63 | 4.53 | 4.61 | 4.60 | 4.62 | 2.30 |
| gpt-oss:120b | low | incubator | 160 | 3.25 | 3.26 | 3.28 | 3.25 | 9.27 | 6.00 |
| gpt-oss:120b | medium | bikeshare | 240 | 4.29 | 4.16 | 3.96 | 3.99 | 4.07 | 2.30 |
| gpt-oss:120b | medium | incubator | 160 | 4.76 | 4.77 | 4.60 | 4.76 | 9.48 | 6.00 |
| gpt-oss:20b | high | bikeshare | 48 | 3.34 | 3.27 | 3.13 | 3.13 | 3.13 | 2.30 |
| gpt-oss:20b | high | incubator | 32 | 5.71 | 5.71 | 5.84 | 5.71 | 7.60 | 6.00 |
| gpt-oss:20b | low | bikeshare | 144 | 3.41 | 3.31 | 3.03 | 3.02 | 3.02 | 2.30 |
| gpt-oss:20b | low | incubator | 96 | 4.12 | 4.12 | 4.08 | 4.12 | 10.02 | 6.00 |
| gpt-oss:20b | medium | bikeshare | 144 | 4.20 | 4.01 | 4.06 | 4.04 | 4.14 | 2.30 |
| gpt-oss:20b | medium | incubator | 96 | 8.02 | 8.02 | 8.02 | 8.02 | 8.99 | 6.00 |
| qwen3.5:9b | none | bikeshare | 240 | 4.93 | 4.37 | 4.59 | 4.88 | 4.60 | 2.30 |
| qwen3.5:9b | none | incubator | 160 | 3.50 | 3.02 | 3.50 | 3.50 | 11.57 | 6.00 |
| no reading: {'T_room': {'low': -20, 'high': 50}} | | incubator | 32 | 7.18 | | | | | 6.00 |
| no reading: {'rent_factor': {'low': 0, 'high': 4}, 'return_factor': {'low': 0, 'high': 4}} | | bikeshare | 48 | 7.43 | | | | | 2.30 |

## B. Sensitivity (registered): mean extra, told / not told, references moved down, as written, up

| model | reasoning | reply | twin | cells | down | as written | up |
|---|---|---|---|---|---|---|---|
| gpt-oss:120b | high | number | bikeshare | 240 | 3.47 / 2.07 | 3.78 / 2.30 | 4.04 / 2.46 |
| gpt-oss:120b | high | number | incubator | 160 | 8.89 / 5.87 | 9.41 / 6.00 | 11.50 / 5.41 |
| gpt-oss:120b | high | range | bikeshare | 240 | 2.82 / 2.07 | 3.09 / 2.30 | 3.29 / 2.46 |
| gpt-oss:120b | high | range | incubator | 160 | 9.02 / 5.87 | 9.49 / 6.00 | 10.38 / 5.41 |
| gpt-oss:120b | low | number | bikeshare | 240 | 3.04 / 2.07 | 3.31 / 2.30 | 3.49 / 2.46 |
| gpt-oss:120b | low | number | incubator | 160 | 5.90 / 5.87 | 6.51 / 6.00 | 8.53 / 5.41 |
| gpt-oss:120b | low | range | bikeshare | 240 | 4.38 / 2.07 | 4.63 / 2.30 | 4.79 / 2.46 |
| gpt-oss:120b | low | range | incubator | 160 | 3.68 / 5.87 | 3.25 / 6.00 | 3.38 / 5.41 |
| gpt-oss:120b | medium | number | bikeshare | 240 | 3.88 / 2.07 | 4.21 / 2.30 | 4.47 / 2.46 |
| gpt-oss:120b | medium | number | incubator | 160 | 8.68 / 5.87 | 9.47 / 6.00 | 11.75 / 5.41 |
| gpt-oss:120b | medium | range | bikeshare | 240 | 4.01 / 2.07 | 4.29 / 2.30 | 4.50 / 2.46 |
| gpt-oss:120b | medium | range | incubator | 160 | 5.10 / 5.87 | 4.76 / 6.00 | 5.12 / 5.41 |
| gpt-oss:20b | high | number | bikeshare | 144 | 2.33 / 2.07 | 2.55 / 2.30 | 2.69 / 2.46 |
| gpt-oss:20b | high | number | incubator | 96 | 9.44 / 5.87 | 9.78 / 6.00 | 11.45 / 5.41 |
| gpt-oss:20b | high | range | bikeshare | 48 | 3.10 / 2.07 | 3.34 / 2.30 | 3.48 / 2.46 |
| gpt-oss:20b | high | range | incubator | 32 | 5.92 / 5.87 | 5.71 / 6.00 | 6.40 / 5.41 |
| gpt-oss:20b | low | number | bikeshare | 144 | 2.25 / 2.07 | 2.45 / 2.30 | 2.57 / 2.46 |
| gpt-oss:20b | low | number | incubator | 96 | 5.73 / 5.87 | 6.17 / 6.00 | 7.05 / 5.41 |
| gpt-oss:20b | low | range | bikeshare | 144 | 3.19 / 2.07 | 3.41 / 2.30 | 3.56 / 2.46 |
| gpt-oss:20b | low | range | incubator | 96 | 4.89 / 5.87 | 4.12 / 6.00 | 4.06 / 5.41 |
| gpt-oss:20b | medium | number | bikeshare | 144 | 3.07 / 2.07 | 3.25 / 2.30 | 3.41 / 2.46 |
| gpt-oss:20b | medium | number | incubator | 96 | 8.60 / 5.87 | 9.00 / 6.00 | 10.37 / 5.41 |
| gpt-oss:20b | medium | range | bikeshare | 144 | 3.96 / 2.07 | 4.20 / 2.30 | 4.39 / 2.46 |
| gpt-oss:20b | medium | range | incubator | 96 | 8.32 / 5.87 | 8.02 / 6.00 | 8.54 / 5.41 |
| qwen3.5:9b | none | number | bikeshare | 240 | 4.98 / 2.07 | 5.24 / 2.30 | 5.41 / 2.46 |
| qwen3.5:9b | none | number | incubator | 160 | 14.39 / 5.87 | 14.61 / 6.00 | 15.37 / 5.41 |
| qwen3.5:9b | none | range | bikeshare | 240 | 4.79 / 2.07 | 4.93 / 2.30 | 5.12 / 2.46 |
| qwen3.5:9b | none | range | incubator | 160 | 3.29 / 5.87 | 3.50 / 6.00 | 3.82 / 5.41 |
