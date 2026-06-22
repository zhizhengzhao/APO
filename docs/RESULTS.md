# APO 六实验最终报告：No-bonus vs Bonus-fixed

生成日期：2026-06-22

本报告基于 `results/summary/apo_six_epoch_summary.json` 与 `apo_six_experiment_data.json` 自动整理，覆盖 3 个任务类别 × 2 种 reward 变体共 6 个训练实验。训练曲线和主要指标均按 epoch 聚合；最终指标使用第 18 个 epoch（steps 425–449）的均值。

## 1. 实验矩阵与统计口径

每个实验训练 18 epochs。每个 epoch 对应 25 个 GRPO steps；单 step 是 mini-batch 口径，波动较大，因此本报告不比较单个 step，只比较 epoch 级均值。

- `no-bonus`：`cost_bonus_scale=0`。
- `bonus-fixed`：`cost_bonus_scale=0.5`，bonus 只在 mixed group 的正确样本内部按 `n_calls` 起 tie-break；全对组 advantage 为 0。

| 类别 | no-bonus | bonus-fixed |
|---|---|---|
| Code（代码） | `code_nobonus` | `code_bonus_fixed` |
| Math（数学） | `math_nobonus` | `math_bonus_fixed` |
| Reasoning（推理） | `reasoning_nobonus` | `reasoning_bonus_fixed` |

## 2. 完成状态与健康检查

6 个实验均完成 18 epochs / 450 GRPO steps，并均生成 `head_grpo_step450`。`details.jsonl` 坏行均为 0，第 18 epoch 的 API error 均为 0。

| Run | epochs | steps | ckpt450 | epoch18 correct | epoch18 n_calls | epoch18 n_active | epoch18 api | bad_json |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `code_nobonus` | 18 | 450 | yes | 0.832 | 8.99 | 4.93 | 0 | 0 |
| `code_bonus_fixed` | 18 | 450 | yes | 0.829 | 7.52 | 3.99 | 0 | 0 |
| `math_nobonus` | 18 | 450 | yes | 0.724 | 8.39 | 4.07 | 0 | 0 |
| `math_bonus_fixed` | 18 | 450 | yes | 0.712 | 7.59 | 3.56 | 0 | 0 |
| `reasoning_nobonus` | 18 | 450 | yes | 0.708 | 6.66 | 3.97 | 0 | 0 |
| `reasoning_bonus_fixed` | 18 | 450 | yes | 0.740 | 4.59 | 2.91 | 0 | 0 |

## 3. Epoch 级训练曲线

![Epoch curves](results/figures/fig4_six_epoch_curves.png)

图中每个点为一个 epoch 内 25 个 step 的均值。

## 4. 第 18 epoch：正确率与调用数

![Epoch 18 correct](results/figures/fig5_epoch18_correct.png)

| 类别 | no-bonus correct | bonus-fixed correct | Δ correct | no-bonus calls | bonus-fixed calls | Δ calls |
|---|---:|---:|---:|---:|---:|---:|
| Code（代码） | 0.832 | 0.829 | -0.003 | 8.99 | 7.52 | -1.47 |
| Math（数学） | 0.724 | 0.712 | -0.011 | 8.39 | 7.59 | -0.81 |
| Reasoning（推理） | 0.708 | 0.740 | +0.031 | 6.66 | 4.59 | -2.07 |

按 epoch-18 均值，bonus-fixed 在三类中均降低平均 `n_calls`：code 约 -1.47，math 约 -0.80，reasoning 约 -2.07。

## 5. Final-30 架构分布

以下架构分布仍使用最终 30 个 step 的采样架构，因为架构 family、role、model 等离散统计需要较大的样本量。

### 5.1 模型选择

![Model share](results/figures/fig2_six_model_share.png)

| Run | final-30 model share (flash / plus / max) |
|---|---|
| `code_nobonus` | flash 12.1% / plus 17.9% / max 69.9% |
| `code_bonus_fixed` | flash 8.1% / plus 17.7% / max 74.2% |
| `math_nobonus` | flash 18.6% / plus 43.5% / max 37.9% |
| `math_bonus_fixed` | flash 19.1% / plus 40.3% / max 40.6% |
| `reasoning_nobonus` | flash 32.7% / plus 51.2% / max 16.1% |
| `reasoning_bonus_fixed` | flash 15.6% / plus 72.5% / max 11.9% |

### 5.2 Active agent 与通信边

![Active agents and edge counts](results/figures/fig3_six_active_edges.png)

| Run | n_active 主分布 | edges_count 主分布 |
|---|---|---|
| `code_nobonus` | 5个 42.9%；6个 28.1%；4个 22.7%；3个 6.0% | 17条 40.2%；26条 26.8%；10条 21.3%；5条 4.7% |
| `code_bonus_fixed` | 4个 49.0%；3个 25.5%；5个 20.8%；6个 3.2% | 8条 34.5%；4条 24.2%；16条 12.3%；14条 8.3% |
| `math_nobonus` | 4个 41.5%；5个 29.1%；3个 21.1%；2个 4.4% | 8条 18.8%；15条 16.2%；4条 13.3%；9条 13.0% |
| `math_bonus_fixed` | 4个 36.3%；3个 35.1%；5个 14.5%；2个 11.9% | 4条 17.6%；5条 11.3%；2条 10.3%；8条 10.1% |
| `reasoning_nobonus` | 4个 40.1%；3个 25.1%；5个 23.8%；2个 5.7% | 4条 22.5%；1条 17.9%；5条 10.0%；10条 6.9% |
| `reasoning_bonus_fixed` | 3个 45.3%；2个 31.0%；4个 19.9%；5个 2.4% | 1条 30.0%；4条 27.6%；5条 17.1%；10条 8.8% |

### 5.3 Role 组成

| Run | final-30 top roles |
|---|---|
| `code_nobonus` | Solver 49.0%；Refiner 28.6%；Verifier 14.5%；Expert 2.3%；Critic 1.8% |
| `code_bonus_fixed` | Solver 70.4%；Refiner 17.5%；Verifier 6.3%；Expert 1.9%；Planner 1.6% |
| `math_nobonus` | Solver 44.3%；Verifier 18.9%；Refiner 13.9%；Tester 7.1%；Critic 5.0% |
| `math_bonus_fixed` | Solver 47.7%；Refiner 16.8%；Verifier 16.0%；Tester 6.0%；Critic 4.8% |
| `reasoning_nobonus` | Solver 39.8%；Verifier 18.9%；Refiner 14.8%；Expert 8.7%；Tester 6.5% |
| `reasoning_bonus_fixed` | Solver 68.4%；Refiner 15.1%；Verifier 12.0%；Critic 1.8%；Tester 1.2% |

## 6. 最常见 architecture family

### `code_nobonus`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Refiner[max] + Solver[max] + Solver[max] + Solver[max]` | 4 | 10 | 29 |
| `Refiner[max] + Solver[max] + Solver[max] + Solver[max] + Verifier[max]` | 5 | 17 | 23 |
| `Refiner[max] + Refiner[plus] + Solver[max] + Solver[max] + Solver[max]` | 5 | 17 | 22 |
| `Refiner[max] + Refiner[max] + Solver[max] + Solver[max] + Solver[max]` | 5 | 17 | 20 |
| `Refiner[max] + Solver[max] + Solver[max] + Verifier[max]` | 4 | 10 | 19 |

### `code_bonus_fixed`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[max] + Solver[max] + Solver[max]` | 3 | 4 | 151 |
| `Solver[max] + Solver[max] + Solver[plus]` | 3 | 4 | 94 |
| `Refiner[max] + Solver[max] + Solver[max] + Solver[max]` | 4 | 8 | 61 |
| `Solver[max] + Solver[max] + Solver[max] + Solver[max]` | 4 | 8 | 45 |
| `Solver[flash] + Solver[max] + Solver[max]` | 3 | 4 | 43 |

### `math_nobonus`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[max] + Solver[max] + Solver[plus]` | 3 | 4 | 8 |
| `Solver[flash] + Solver[max] + Solver[plus]` | 3 | 4 | 7 |
| `Solver[max] + Solver[plus] + Verifier[plus]` | 3 | 4 | 6 |
| `Solver[flash] + Solver[plus] + Verifier[plus]` | 3 | 4 | 6 |
| `Solver[max] + Solver[plus] + Solver[plus]` | 3 | 4 | 6 |

### `math_bonus_fixed`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[max] + Solver[plus]` | 2 | 2 | 28 |
| `Solver[max] + Solver[max]` | 2 | 2 | 15 |
| `Solver[plus] + Solver[plus]` | 2 | 2 | 14 |
| `Refiner[max] + Solver[max] + Solver[plus]` | 3 | 4 | 12 |
| `Solver[max] + Solver[plus] + Verifier[max]` | 3 | 4 | 10 |

### `reasoning_nobonus`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[flash] + Solver[plus] + Verifier[plus]` | 3 | 1 | 17 |
| `Solver[flash] + Solver[plus]` | 2 | 0 | 10 |
| `Solver[flash] + Solver[plus] + Solver[plus]` | 3 | 1 | 10 |
| `Solver[flash] + Solver[max] + Solver[plus]` | 3 | 1 | 8 |
| `Solver[flash] + Solver[flash] + Solver[plus]` | 3 | 1 | 7 |

### `reasoning_bonus_fixed`

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[plus] + Solver[plus]` | 2 | 1 | 405 |
| `Solver[flash] + Solver[plus]` | 2 | 1 | 60 |
| `Solver[plus] + Solver[plus] + Verifier[flash]` | 3 | 4 | 59 |
| `Solver[plus] + Solver[plus] + Verifier[plus]` | 3 | 4 | 52 |
| `Solver[plus] + Solver[plus] + Verifier[max]` | 3 | 4 | 50 |

## 7. 复核路径

| 内容 | 路径 |
|---|---|
| epoch 级统计 | `results/summary/apo_six_epoch_summary.json` |
| 6 实验 summary | `results/summary/apo_six_experiment_data.json` |
| 每个实验 history | `results/<run>/history.json` |
| 每个实验 per-trace details | `results/<run>/details.jsonl` |
| 代码 | `src/arch_policy/` |
| 数据 | `data/categories/` |
| 图 | `results/figures/` |

## 8. 口径说明

- 本报告的主要结果按 epoch 聚合；不比较单个 step。
- 本报告使用 training/on-policy 指标，不包含 held-out test evaluation。
- `details.jsonl` 记录 `edges_count`，不保存完整 edge matrix。
- 训练 checkpoint 未放入 GitHub repo，但服务器上每个实验均保留 `head_grpo_step450` 与中间每 25 step 的 checkpoint。
