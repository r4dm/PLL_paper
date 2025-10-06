PY=python

.PHONY: setup warmup pac pla-train pla-eval report geometry piecewise

setup:
	$(PY) -m pip install -e .

warmup:
	@echo "ok" > runs/warmup_ok.txt

pac:
	$(PY) -m pll.pac_audit \
		--texts-file data/sem_mask_train.jsonl \
		--max-texts 500 \
		--phase-layers 0 \
		--weight-layers 0 \
		--decoder models/decoder_v2 \
		--out runs/apc_before.json \
		--n 7 --beta 0.35 --baseline-trials 1000 --per-head

pla-train:
	$(PY) -m pll.train_pla --pla_layer 0 --pla_part o_proj --pla_heads 8 --pla_n 7 --pla_beta 0.35 --pla_lambda 0.15 --pla_mu 0.02 --texts_file data/sem_mask_train.jsonl --decoder models/decoder_v2 --outdir runs/pla

pla-eval:
	$(PY) -m pll.eval_pla_apc --texts_file data/sem_mask_train.jsonl --layer 0 --head 8 --n 7 --beta 0.35 --decoder models/decoder_v2 --out runs/apc_after.json

report:
	$(PY) scripts/make_report.py

geometry:
	$(PY) scripts/geometry_probe.py --layers 0,6,11 --max-n 16 --device auto --out runs/geometry_probe_full_0_6_11.json --baseline-trials 100 --per-head
	$(PY) scripts/make_geometry_figs.py --in runs/geometry_probe_full_0_6_11.json --out figs/geometry_summary.csv
	$(PY) scripts/render_geometry_plots.py --csv figs/geometry_summary.csv --out-dir figs

piecewise:
	$(PY) scripts/extract_piecewise_sps.py --layer 0 --part mlp_gate --dataset data/sem_mask_train.jsonl --out runs/piecewise_sps_l0_mlp_gate.json
	$(PY) scripts/extract_piecewise_sps.py --layer 0 --part mlp_up --dataset data/sem_mask_train.jsonl --out runs/piecewise_sps_l0_mlp_up.json
	$(PY) scripts/extract_piecewise_sps.py --layer 0 --part mlp_down --dataset data/sem_mask_train.jsonl --out runs/piecewise_sps_l0_mlp_down.json
	$(PY) scripts/make_piecewise_psps_figs.py --in runs/piecewise_sps_l0_mlp_gate.json runs/piecewise_sps_l0_mlp_up.json runs/piecewise_sps_l0_mlp_down.json --out figs/piecewise_psps_summary.csv --top 20
	$(PY) scripts/render_psps_plots.py --summary figs/piecewise_psps_summary.csv --tiles figs/piecewise_psps_tiles.csv --out-dir figs
