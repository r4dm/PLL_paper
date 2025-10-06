# PLL Paper Artifact (Minimal)

This repository contains the implementation and experimental artifacts for the paper **"Bridge: PLL Algebra in LLMs — From Geometry to Behavior"** ([PAPER.md](PAPER.md)).

## Project Overview

We study whether internal weights and activations of contemporary LLMs exhibit a low-dimensional geometric structure consistent with a finite phase-lattice logic (PLL) and whether this structure can be used to causally and safely adjust behavior. Our work introduces:

- **PAC (Phase Alignment Consistency)**: A token-level metric connecting token phases to phase-lattice rays
- **PLA (Phase-Lattice Adapter)**: A lightweight per-head projection that softly nudges activations toward nearest phase rays
- **Piecewise-SPS (PSPS)**: Extraction of compact symbolic phase sum formulas from MLP regions

## Key Findings

On Qwen3-0.6B we find that 2D projections of attention/MLP weights align with cyclic phase lattices (Cn) with coverage ≈ 1.0 and strong phase-complementarity indicators. PLA yields consistent increases in PAC (up to +0.02) without degrading language quality (ΔPPL ≤ +0.5% on WT2/internal). Finally, we extract compact Piecewise-SPS formulas (≤ 5 terms, R² ≥ 0.9) for layer-0 MLP regions, providing symbolic evidence of local linear laws.

For detailed methodology, results, and discussion, see [PAPER.md](PAPER.md).

Quickstart

```bash
# 1) Install
pip install -e .

# 2) Geometry probe + figures
python scripts/geometry_probe.py \
  --layers 0,6,11 --max-n 16 --device auto \
  --out runs/geometry_probe_full_0_6_11.json --baseline-trials 100 --per-head
python scripts/make_geometry_figs.py \
  --in runs/geometry_probe_full_0_6_11.json \
  --out figs/geometry_summary.csv
python scripts/render_geometry_plots.py \
  --csv figs/geometry_summary.csv --out-dir figs

# 3) PAC audit (decoder mode, phase layer 0)
python -m pll.pac_audit \
  --texts-file data/sem_mask_train.jsonl --max-texts 500 \
  --phase-layers 0 \
  --weight-layers 0 \
  --decoder models/decoder_v2 \
  --out runs/apc_before.json \
  --n 7 --beta 0.35 --baseline-trials 1000 --per-head

# 4) Train PLA (o_proj.h8)
python -m pll.train_pla \
  --pla_layer 0 --pla_part o_proj --pla_heads 8 \
  --pla_n 7 --pla_beta 0.35 --pla_lambda 0.15 --pla_mu 0.02 \
  --texts_file data/sem_mask_train.jsonl \
  --decoder models/decoder_v2 \
  --outdir runs/pla

# 5) Evaluate APC after PLA
python -m pll.eval_pla_apc \
  --texts_file data/sem_mask_train.jsonl \
  --layer 0 --head 8 --n 7 --beta 0.35 \
  --decoder models/decoder_v2 \
  --out runs/apc_after.json

# 6) Piecewise-SPS extraction (layer 0 MLP)
python scripts/extract_piecewise_sps.py \
  --layer 0 --part mlp_gate --dataset data/sem_mask_train.jsonl \
  --out runs/piecewise_sps_l0_mlp_gate.json
python scripts/extract_piecewise_sps.py \
  --layer 0 --part mlp_up --dataset data/sem_mask_train.jsonl \
  --out runs/piecewise_sps_l0_mlp_up.json
python scripts/extract_piecewise_sps.py \
  --layer 0 --part mlp_down --dataset data/sem_mask_train.jsonl \
  --out runs/piecewise_sps_l0_mlp_down.json
python scripts/make_piecewise_psps_figs.py \
  --in runs/piecewise_sps_l0_mlp_gate.json \
      runs/piecewise_sps_l0_mlp_up.json \
      runs/piecewise_sps_l0_mlp_down.json \
  --out figs/piecewise_psps_summary.csv --top 20
python scripts/render_psps_plots.py \
  --summary figs/piecewise_psps_summary.csv \
  --tiles figs/piecewise_psps_tiles.csv --out-dir figs

# 7) Optional: aggregate PAC / PLA figures
python scripts/make_pac_figs.py --in runs/apc_before.json runs/apc_after.json \
  runs/pla/apc_before_after.json --out figs/pac_summary.csv
python scripts/render_pac_plots.py --csv figs/pac_summary.csv --out-dir figs --top 20
python scripts/make_pla_ablation_figs.py --in runs/pla runs/pla_l0_* \
  --pac-baseline runs/apc_before.json --out figs/pla_ablation.csv
python scripts/render_pla_plots.py --csv figs/pla_ablation.csv --out figs/pla_delta_apc.png
```

> Makefile shortcuts: `make geometry`, `make pac`, `make pla-train`, `make pla-eval`,
> `make piecewise`, `make report`.

Scripts
- `geometry_probe.py`: phase-lattice geometry/DPN probe, writes JSON reports.
- `make_*.py`: convert JSON outputs (geometry, PAC, PLA, PSPS) into CSV summaries.
- `render_*.py`: render Matplotlib figures from the CSVs.
- `extract_piecewise_sps.py`: run the PSPS regionizer/fit/export pipeline.
- Example PLA runs from the paper are included under `runs/pla_l0_*`; add your
  own directories there to compare alternative ablations.

Repo layout
- `data/`: small text snippets and masks used by the quickstart scripts.
- `models/decoder_v2/`: lightweight decoder checkpoint (config + weights).
- `models/Qwen3-0.6B`: place the base model here (or point `PLL_MODEL_DIR` at another path).
- `runs/`: outputs from geometry/PAC/PLA/PSPS pipelines (JSON/CSV/markdown).
- `figs/`: rendered PNG/CSV artifacts referenced in the paper.
- `src/pll/`: the runnable PLL modules (runtime helpers, PAC/PLA/PSPS code).

Environment
- Python >= 3.10
- Torch and Transformers are required (pyproject lists versions)
- Set env `PLL_MODEL_DIR` to point to a local Hugging Face checkpoint directory
  (offline mode). Fallback: set `PLL_MODEL` to a public HF model id.
- Device selection defaults to CUDA → MPS → CPU when `--device auto`.
