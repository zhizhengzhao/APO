# 结果与方法（measured facts）

本文档只陈述测得的量与图的内容，不含结论性判断；论文叙述与结论由作者撰写。所有数字来自 `results/summary/apo_three_experiment_data.json`，统计窗口为最终 30 步的架构样本与最终 20 步的训练指标。

## 1. 方法概述

策略 head 在给定任务后采样一个多智能体架构 A，包含：激活的 agent 数量、每个 agent 的 role、每个 agent 的 worker 模型（Qwen 三档之一）、执行顺序、有向通信边。底层 worker pool 固定，仅训练 head：先在共享 SFT 池上做 SFT 暖启动，再按类别用 GRPO 训练。三类共用同一 SFT 起点与同一 reward shaping，仅按类别的任务 reward 不同。

通信语义：边为有向 `edges[i,j]`，agent j 能直接看到 agent i 当且仅当存在直接边 i→j；可见性不传递（A→B、B→C 不蕴含 C 可见 A，除非另有直接边 A→C）。`details.jsonl` 记录每个架构的 `edges_count`，未记录完整 edge 矩阵。

## 2. 最终训练状态

三类均完成 450 步（18 epoch），均生成 `head_grpo_step450`，最后 20 步无 API error。

| 类别 | history 步数 | 最终 ckpt | 初始 acc(前20步) | 最终 acc(后20步) | 最终 n_active | 最终 n_calls |
|---|---:|---:|---:|---:|---:|---:|
| 代码 (code) | 450 | step450 | 67.3% | 77.3% | 2.26 | 4.65 |
| 数学 (math) | 450 | step450 | 70.6% | 72.1% | 2.09 | 5.82 |
| 推理 (reasoning) | 450 | step450 | 64.8% | 73.3% | 2.91 | 4.55 |

## 3. 模型选择（最终 30 步，按 agent slot 计）

| 类别 | flash / plus / max |
|---|---|
| 代码 (code) | flash 30.0% / plus 18.3% / max 51.6% |
| 数学 (math) | flash 22.5% / plus 49.7% / max 27.8% |
| 推理 (reasoning) | flash 15.6% / plus 72.5% / max 11.9% |

## 4. 角色组成（最终 30 步）

| 类别 | 前 4 role 占比 |
|---|---|
| 代码 (code) | Solver 65.9%；Planner 25.2%；Refiner 5.1%；Expert 1.4% |
| 数学 (math) | Solver 45.3%；Expert 17.1%；Verifier 11.5%；Refiner 6.7% |
| 推理 (reasoning) | Solver 68.4%；Refiner 15.1%；Verifier 12.0%；Critic 1.8% |

## 5. Active agent 数与通信边数量（最终 30 步）

| 类别 | n_active 占比(前3) | edges_count 占比(前4) |
|---|---|---|
| 代码 (code) | 2 个 69.8%；3 个 27.8%；1 个 1.8% | 2 条 69.7%；5 条 26.2%；0 条 1.8%；4 条 1.6% |
| 数学 (math) | 2 个 61.4%；3 个 19.5%；1 个 16.5% | 0 条 74.8%；2 条 13.1%；1 条 6.3%；3 条 3.1% |
| 推理 (reasoning) | 3 个 45.3%；2 个 31.0%；4 个 19.9% | 1 条 30.0%；4 条 27.6%；5 条 17.1%；10 条 8.8% |

## 6. 最常见架构 family（最终 30 步）

### 代码 (code)

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Planner[flash] + Solver[max]` | 2 | 2 | 400 |
| `Solver[flash] + Solver[max]` | 2 | 2 | 213 |
| `Planner[max] + Solver[max]` | 2 | 2 | 117 |
| `Solver[max] + Solver[max]` | 2 | 2 | 93 |

### 数学 (math)

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[plus] + Solver[plus]` | 2 | 0 | 80 |
| `Solver[plus]` | 1 | 0 | 73 |
| `Solver[max] + Solver[plus]` | 2 | 0 | 72 |
| `Solver[flash] + Solver[plus]` | 2 | 0 | 65 |

### 推理 (reasoning)

| family | n_active | edges | count |
|---|---:|---:|---:|
| `Solver[plus] + Solver[plus]` | 2 | 1 | 405 |
| `Solver[flash] + Solver[plus]` | 2 | 1 | 60 |
| `Solver[plus] + Solver[plus] + Verifier[flash]` | 3 | 4 | 59 |
| `Solver[plus] + Solver[plus] + Verifier[plus]` | 3 | 4 | 52 |

## 7. 图

| 图 | 内容 |
|---|---|
| `results/figures/fig1_training_curves.png` | accuracy / n_calls / n_active / entropy 随步数变化（15 步滑动平均） |
| `results/figures/fig2_model_share.png` | 三类最终模型选择比例 |
| `results/figures/fig3_role_share.png` | 三类最终 role 组成比例 |
| `results/figures/fig4_active_edges.png` | 三类 n_active 分布与通信边数量分布 |
| `results/figures/fig5_role_model_heatmap.png` | 三类 role→model 偏好热图 |
| `results/figures/fig6_typical_architecture_families.png` | 三类最常见架构 family 的示意图（family 级，非精确 edge 矩阵） |

## 8. 数据与口径说明

- 训练 acc 为 on-policy 训练 batch 的 correct rate，非固定 held-out 测试集曲线；本 repo 未包含 held-out 评估或固定架构 baseline 网格（baseline 定义见 `src/arch_policy/baselines.py`，评估入口 `scripts/04_evaluate.py`）。
- 通信仅报告 `edges_count`（可靠），未报告精确每条边方向（最终日志未保存 edge 矩阵）。
- 三类共用同一 reward shaping（`src/arch_policy/training/advantage.py`）；模型权重未随 repo 提供。
