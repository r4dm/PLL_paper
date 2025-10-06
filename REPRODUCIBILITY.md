# Reproducibility

Steps
- make setup — install dependencies
- make warmup — check imports, create minimal model config
- make geometry — weight geometry + CSV/figures (runs/geometry_*.json, figs/geometry_*)
- make pac — decoder PAC on `sem_mask_train.jsonl` (500 texts, phase layer 0, 1000 random-φ trials) → runs/apc_before.json
- make pla-train — write training summary PLA to runs/pla/
- make pla-eval — calculate PAC after, runs/apc_after.json
- make piecewise — Piecewise-SPS reports (runs/piecewise_sps_*.json) and PNG/CSV
- make report — report runs/report.md

Environment
- Python >= 3.10, macOS/Linux
- Before running, set `PLL_MODEL_DIR` to local model directory
  (or `PLL_MODEL` to public id). After first download, Hugging Face
  uses local cache.
- Determinism: fixed seed in CLI not required for stubs
