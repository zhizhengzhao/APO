# Six-run measured results

This document reports measured quantities only. All values are parsed from `results/summary/apo_six_experiment_data.json`.

## Final training status

| Run | steps | correct_last20 | calls_last20 | active_last20 | api_last20 | bad_json |
|---|---:|---:|---:|---:|---:|---:|
| `code_nobonus` | 450 | 0.821 | 9.02 | 4.93 | 0 | 0 |
| `code_bonus_fixed` | 450 | 0.822 | 7.51 | 3.98 | 0 | 0 |
| `math_nobonus` | 450 | 0.725 | 8.30 | 4.08 | 0 | 0 |
| `math_bonus_fixed` | 450 | 0.708 | 7.46 | 3.54 | 0 | 0 |
| `reasoning_nobonus` | 450 | 0.704 | 6.63 | 3.99 | 0 | 0 |
| `reasoning_bonus_fixed` | 450 | 0.733 | 4.55 | 2.91 | 0 | 0 |

## Reward comparison by category

| Category | no-bonus correct | bonus-fixed correct | no-bonus calls | bonus-fixed calls |
|---|---:|---:|---:|---:|
| code | 0.821 | 0.822 | 9.02 | 7.51 |
| math | 0.725 | 0.708 | 8.30 | 7.46 |
| reasoning | 0.704 | 0.733 | 6.63 | 4.55 |

## Top architecture family in final 30 steps

| Run | family | n_active | edges | count |
|---|---|---:|---:|---:|
| `code_nobonus` | `Refiner[max] + Solver[max] + Solver[max] + Solver[max]` | 4 | 10 | 29 |
| `code_bonus_fixed` | `Solver[max] + Solver[max] + Solver[max]` | 3 | 4 | 151 |
| `math_nobonus` | `Solver[max] + Solver[max] + Solver[plus]` | 3 | 4 | 8 |
| `math_bonus_fixed` | `Solver[max] + Solver[plus]` | 2 | 2 | 28 |
| `reasoning_nobonus` | `Solver[flash] + Solver[plus] + Verifier[plus]` | 3 | 1 | 17 |
| `reasoning_bonus_fixed` | `Solver[plus] + Solver[plus]` | 2 | 1 | 405 |

## Figures

- `results/figures/fig1_six_acc_calls.png`: final-20 correct rate and mean LLM calls.
- `results/figures/fig2_six_model_share.png`: final-30 model share.
- `results/figures/fig3_six_active_edges.png`: final-30 active-agent and edge-count distributions.
