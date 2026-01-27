# Paper Results Index

This file pins the key experimental artifacts for citation and review.  
All results correspond to tag: `emst-paper-20260124` (see `RELEASE_NOTES_v0.1-paper.md`).

## Core tables
- A0/A1/A2 table (Geo/Band, mean±std): `stats/ablation_a012_table.tsv`
- A0/A1/A2 significance (paired bootstrap/t-test): `checkpoints/ablation_a012_stats.json`
- Run-based significance summary: `stats/summary.tsv`
- Geo heuristic baseline: `stats/baseline_heuristic_metrics.json`
- Event.x-only MLP baseline: `stats/baseline_mlp_metrics.json`
- prev_event k sweep (A2, seed=0): `stats/prev_event_k_sweep.tsv`

## Core figures
- Geo ablation error bars: `figures/geo_ablation_mrr.png`
- Band B1 (mask band_obs) comparison: `figures/band_b1_compare.png`
- Reliability diagram (Geo/Band): `figures/reliability_diagram.png`
- Band difficulty curve: `figures/band_difficulty_curve.png`

## Key metrics snapshots
- Default deterministic metrics: `stats/default_metrics.json`
- Default calibrated (TS) metrics: `stats/default_calib_metrics.json`
- Default MC Dropout metrics: `stats/default_mc_metrics.json`
- B1 deterministic metrics: `stats/b1_metrics.json`
- Band difficulty table: `band_difficulty/band_difficulty.tsv`
 - Band calibration table (d1/d3/d5): `band_difficulty/band_calib.tsv`

## Dataset note
每个事件默认仅由一个传感器观测（`observed_by` 度恒为 1），这是设计选择而非 bug。

## Baselines (test)
- Heuristic Geo (sensor geocell distance rank): MRR 0.1076, Hits@10 0.2570  
- MLP (event.x only): Geo MRR 0.0254, Hits@10 0.0431; Band MRR 0.4984, Hits@10 1.0000

## prev_event k sweep (A2, seed=0)
`stats/prev_event_k_sweep.tsv`  
- k=1: Geo MRR 0.1083, Hits@10 0.2524  
- k=4: Geo MRR 0.1161, Hits@10 0.2816  
- k=8: Geo MRR 0.1273, Hits@10 0.2977  
- k=16: Geo MRR 0.1134, Hits@10 0.2723

## Calibration & Uncertainty (实验小节)
### Uncalibrated vs Temperature Scaling (TS)
**Test split (default config)**  
- Geo NLL/ECE: **4.1738 / 0.0083** → **4.1248 / 0.0278** (TS)  
- Band NLL/ECE: **1.4787 / 0.1710** → **1.2455 / 0.0577** (TS)  

TS 对 Band 显著降低 NLL/ECE；Geo 的 NLL 小幅下降，但 ECE 上升（温度 < 1 使分布更尖锐），说明 Geo 任务仍偏弱置信。

### MC Dropout 不确定性（test）
**Entropy (mean / p50 / p90 / p99)**  
- Geo: 4.146 / 4.156 / 4.456 / 4.807  
- Band: 1.264 / 1.286 / 1.525 / 1.670  

**Variance (mean / p50 / p90 / p99)**  
- Geo: 4.47e‑05 / 4.06e‑05 / 6.92e‑05 / 1.18e‑04  
- Band: 7.38e‑03 / 7.34e‑03 / 9.12e‑03 / 1.07e‑02  

**Risk‑Coverage (test)**  
- Geo RC@{1.0/0.8/0.6/0.4}: 0.0306 / 0.0324 / 0.0351 / 0.0387  
- Band RC@{1.0/0.8/0.6/0.4}: 0.4309 / 0.4536 / 0.4797 / 0.4944  

解释：Geo 的选择性收益有限；Band 在高置信覆盖下有更明显提升。

### 噪声难度与校准收益（d1/d3/d5）
`band_difficulty/band_calib.tsv` 显示三档难度均获得 NLL/ECE 改善：  
- d1: NLL 1.4856→1.2389, ECE 0.1600→0.0882  
- d3: NLL 1.9327→1.8299, ECE 0.0779→0.0458  
- d5: NLL 2.2040→2.1772, ECE 0.0615→0.0294  

结论：校准在不同噪声强度下稳定改善可信度，但收益幅度随任务难度变化而不同。

## Δt supplemental
- Δt encoding ablations (seed=0):  
  - log encoding: `checkpoints/dt_log_seed0.pt` (see eval logs)  
  - sin/cos encoding: `checkpoints/dt_sincos_seed0.pt` (see eval logs)
- Strong temporal controls (seed=0): configs `configs/strong_time.yaml`, `configs/strong_time_v2.yaml`  
  (A1/A2 deltas reported in experiment logs; see `PROJECT_PROPOSAL.md` appendix notes.)
- Δt log 标准化统计使用 train-only，valid/test 复用训练统计（避免 test-time 适配）。
- Train-only normalization check (seed=0, default): A0 Geo MRR 0.0910, A1 0.1130, A2 0.1150（相对关系保持）。

### Strong-time stability (3 seeds)
Appendix table: `stats/strong_time_seed3.tsv`  
- strong_time: A1 Geo MRR 0.0914±0.0080 vs A2 0.0921±0.0092  
- strong_time_v2: A1 Geo MRR 0.1142±0.0113 vs A2 0.1138±0.0094  
结论：Δt 贡献依然边际，稳定性结论成立。
