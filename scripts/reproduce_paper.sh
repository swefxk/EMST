#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/paper_run/${RUN_ID}"

mkdir -p "$OUT_DIR"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n emst_kgc python)
else
  PYTHON_CMD=(python)
fi

run_py() {
  "${PYTHON_CMD[@]}" "$@"
}

echo "Paper run: ${RUN_ID}" | tee "$OUT_DIR/run_info.txt"
git -C "$ROOT_DIR" rev-parse HEAD >> "$OUT_DIR/run_info.txt"

echo "[1/8] Data generation"
run_py "$ROOT_DIR/data_gen/build_kg_files.py" --config "$ROOT_DIR/configs/default.yaml" --out "$ROOT_DIR/data/synth1"

echo "[2/8] Build PyG HeteroData"
run_py "$ROOT_DIR/pyg_data/build_heterodata.py" --data_dir "$ROOT_DIR/data/synth1" --out "$ROOT_DIR/data/synth1/pyg.pt"

echo "[3/8] Sanity check"
run_py "$ROOT_DIR/sanity_check.py" --data "$ROOT_DIR/data/synth1/pyg.pt" | tee "$OUT_DIR/sanity_check.log"

echo "[4/8] Train default model"
run_py "$ROOT_DIR/train.py" --data "$ROOT_DIR/data/synth1/pyg.pt" --save "$ROOT_DIR/checkpoints/synth1/model.pt"

echo "[5/8] Evaluate default model (deterministic)"
run_py "$ROOT_DIR/eval.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --ckpt "$ROOT_DIR/checkpoints/synth1/model.pt" \
  --out_metrics "$ROOT_DIR/stats/default_metrics.json" | tee "$OUT_DIR/eval_default.log"

echo "[6/8] Calibrate + MC Dropout"
run_py "$ROOT_DIR/calibrate.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --ckpt "$ROOT_DIR/checkpoints/synth1/model.pt" \
  --out "$ROOT_DIR/checkpoints/synth1/calib.json"
run_py "$ROOT_DIR/eval.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --ckpt "$ROOT_DIR/checkpoints/synth1/model.pt" \
  --calib "$ROOT_DIR/checkpoints/synth1/calib.json" \
  --mc_dropout 20 \
  --out_metrics "$ROOT_DIR/stats/default_mc_metrics.json" | tee "$OUT_DIR/eval_mc.log"

echo "[7/8] Train B1 (band_obs masked) and evaluate"
run_py "$ROOT_DIR/train.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --save "$ROOT_DIR/checkpoints/synth1/model_b1.pt" \
  --band_obs_mask
run_py "$ROOT_DIR/eval.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --ckpt "$ROOT_DIR/checkpoints/synth1/model_b1.pt" \
  --band_obs_mask \
  --out_metrics "$ROOT_DIR/stats/b1_metrics.json" | tee "$OUT_DIR/eval_b1.log"

echo "[8/8] A0/A1/A2 ablation table + figures"
run_py "$ROOT_DIR/experiments/ablation_a012.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --config "$ROOT_DIR/configs/default.yaml" \
  --out "$ROOT_DIR/checkpoints/ablation_a012.json"
run_py "$ROOT_DIR/experiments/ablation_stats.py" \
  --input "$ROOT_DIR/checkpoints/ablation_a012.json" \
  --out_json "$ROOT_DIR/checkpoints/ablation_a012_stats.json" \
  --out_table "$ROOT_DIR/stats/ablation_a012_table.tsv"
run_py "$ROOT_DIR/experiments/plot_ablation_geo.py" \
  --input "$ROOT_DIR/checkpoints/ablation_a012.json" \
  --out "$ROOT_DIR/figures/geo_ablation_mrr.png"
run_py "$ROOT_DIR/experiments/plot_band_b1.py" \
  --default_metrics "$ROOT_DIR/stats/default_metrics.json" \
  --b1_metrics "$ROOT_DIR/stats/b1_metrics.json" \
  --out "$ROOT_DIR/figures/band_b1_compare.png"
run_py "$ROOT_DIR/experiments/plot_reliability.py" \
  --data "$ROOT_DIR/data/synth1/pyg.pt" \
  --ckpt "$ROOT_DIR/checkpoints/synth1/model.pt" \
  --calib "$ROOT_DIR/checkpoints/synth1/calib.json" \
  --mc_dropout 20 \
  --prev_event on \
  --split test \
  --out "$ROOT_DIR/figures/reliability_diagram.png"

echo "[copy] Collect artifacts"
mkdir -p "$OUT_DIR/results"
cp "$ROOT_DIR/RESULTS.md" "$OUT_DIR/"
cp "$ROOT_DIR/PROJECT_PROPOSAL.md" "$OUT_DIR/"
cp "$ROOT_DIR/RELEASE_NOTES_v0.1-paper.md" "$OUT_DIR/"
cp "$ROOT_DIR/configs/default.yaml" "$OUT_DIR/"
cp "$ROOT_DIR/stats/ablation_a012_table.tsv" "$OUT_DIR/results/"
cp "$ROOT_DIR/checkpoints/ablation_a012_stats.json" "$OUT_DIR/results/"
cp "$ROOT_DIR/stats/default_metrics.json" "$OUT_DIR/results/"
cp "$ROOT_DIR/stats/default_mc_metrics.json" "$OUT_DIR/results/"
cp "$ROOT_DIR/stats/b1_metrics.json" "$OUT_DIR/results/"
cp "$ROOT_DIR/figures/geo_ablation_mrr.png" "$OUT_DIR/results/"
cp "$ROOT_DIR/figures/band_b1_compare.png" "$OUT_DIR/results/"
cp "$ROOT_DIR/figures/reliability_diagram.png" "$OUT_DIR/results/"
cp "$ROOT_DIR/figures/band_difficulty_curve.png" "$OUT_DIR/results/"
cp "$ROOT_DIR/band_difficulty/band_difficulty.tsv" "$OUT_DIR/results/"

echo "Done. Artifacts in: $OUT_DIR"
