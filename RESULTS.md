# Paper Results Index

This file pins the key experimental artifacts for citation and review.  
All results correspond to tag: `emst-paper-20260124` (see `RELEASE_NOTES_v0.1-paper.md`).

## Core tables
- A0/A1/A2 table (Geo/Band, mean±std): `stats/ablation_a012_table.tsv`
- A0/A1/A2 significance (paired bootstrap/t-test): `checkpoints/ablation_a012_stats.json`
- Run-based significance summary: `stats/summary.tsv`

## Core figures
- Geo ablation error bars: `figures/geo_ablation_mrr.png`
- Band B1 (mask band_obs) comparison: `figures/band_b1_compare.png`
- Reliability diagram (Geo/Band): `figures/reliability_diagram.png`
- Band difficulty curve: `figures/band_difficulty_curve.png`

## Key metrics snapshots
- Default deterministic metrics: `stats/default_metrics.json`
- B1 deterministic metrics: `stats/b1_metrics.json`
- Band difficulty table: `band_difficulty/band_difficulty.tsv`

## Δt supplemental
- Δt encoding ablations (seed=0):  
  - log encoding: `checkpoints/dt_log_seed0.pt` (see eval logs)  
  - sin/cos encoding: `checkpoints/dt_sincos_seed0.pt` (see eval logs)
- Strong temporal controls (seed=0): configs `configs/strong_time.yaml`, `configs/strong_time_v2.yaml`  
  (A1/A2 deltas reported in experiment logs; see `PROJECT_PROPOSAL.md` appendix notes.)
