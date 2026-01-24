# Release Notes: v0.1-paper (emst-paper-20260124)

## Scope
Frozen experimental artifacts for paper submission. All results are traceable to:
- Tag: `emst-paper-20260124`
- Commit: see `git rev-parse emst-paper-20260124`

## Default configuration
- `configs/default.yaml`
- Data size: `T=30000`, `events_per_step=1.2`
- FP rate: `base_p_fp=0.004`
- Time-split: 60/20/20, no leakage

## Core experiments & artifacts
- **A0/A1/A2** (time-split, Geo/Band):
  - Table: `stats/ablation_a012_table.tsv`
  - Significance: `checkpoints/ablation_a012_stats.json`
  - Figure: `figures/geo_ablation_mrr.png`
- **B1 (band_obs masked)**:
  - Metrics: `stats/b1_metrics.json`
  - Figure: `figures/band_b1_compare.png`
- **Calibration & uncertainty**:
  - Reliability diagram: `figures/reliability_diagram.png`
  - Deterministic metrics: `stats/default_metrics.json`
  - TS calibrated metrics: `stats/default_calib_metrics.json`
  - MC Dropout metrics: `stats/default_mc_metrics.json`
- **Band difficulty curve**:
  - Table: `band_difficulty/band_difficulty.tsv`
  - Figure: `figures/band_difficulty_curve.png`
  - Calibration table: `band_difficulty/band_calib.tsv`

## Supplemental (Δt explanations)
- Encoding ablations (seed=0): log/sincos runs (see experiment logs)
- Strong-time controls: `configs/strong_time.yaml`, `configs/strong_time_v2.yaml`

## Repro entrypoint
Run the packaged pipeline and collect artifacts:
```
bash scripts/reproduce_paper.sh
```

Artifacts are written to: `artifacts/paper_run/<timestamp>/`
