# ESM3 -> GNN Distillation Baseline (DISPEF-M)

Baseline-only implementation of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

Implemented scope:

- `sequence + structure -> teacher ESM3 SS8 probabilities -> student GNN`
- student trained on soft teacher targets (plus optional DSSP auxiliary loss)

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Project Layout

```text
/opt/dlami/nvme/esm3_gnn_distill_baseline
├── cache/
├── checkpoints/
├── configs/
│   └── baseline_dispef_m.yaml
├── data/
│   ├── __init__.py
│   ├── constants.py
│   ├── dssp.py
│   ├── graph_builder.py
│   ├── io_utils.py
│   ├── preprocess_dispef.py
│   └── pyg_dataset.py
├── envs/
│   └── environment.yml
├── eval/
│   ├── __init__.py
│   ├── evaluate.py
│   └── metrics.py
├── logs/
├── models/
│   ├── __init__.py
│   ├── factory.py
│   └── gnn.py
├── outputs/
├── scripts/
│   ├── download_dispef.sh
│   ├── eval_baseline.sh
│   ├── generate_teacher_labels.sh
│   ├── preprocess_dispef_m.sh
│   ├── setup_env_nvme.sh
│   └── train_baseline.sh
├── teacher/
│   ├── __init__.py
│   ├── base.py
│   ├── esm3_teacher.py
│   ├── generate_teacher_labels.py
│   └── label_cache.py
├── train/
│   ├── __init__.py
│   ├── config_utils.py
│   ├── losses.py
│   ├── train.py
│   ├── trainer.py
│   └── utils.py
└── requirements.txt
```

## Storage Locations (all under `/opt/dlami/nvme`)

- project: `/opt/dlami/nvme/esm3_gnn_distill_baseline`
- conda env prefix: `/opt/dlami/nvme/envs/esm3_gnn_distill`
- raw DISPEF: `/opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef`
- processed DISPEF-M: `/opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed/dispef_m`
- teacher cache: `/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher/dispef_m`
- checkpoints: `/opt/dlami/nvme/esm3_gnn_distill_baseline/checkpoints`
- run outputs: `/opt/dlami/nvme/esm3_gnn_distill_baseline/outputs`

## 1) Environment Setup (Miniforge at `~/miniforge3`)

```bash
source ~/miniforge3/etc/profile.d/conda.sh

conda env remove -p /opt/dlami/nvme/envs/esm3_gnn_distill -y || true
conda env create -p /opt/dlami/nvme/envs/esm3_gnn_distill -f /opt/dlami/nvme/esm3_gnn_distill_baseline/envs/environment.yml
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torch-geometric
```

Single-command alternative:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/setup_env_nvme.sh
```

## 2) Download and Unpack DISPEF

```bash
mkdir -p /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef
curl -L "https://zenodo.org/api/records/13755810/files-archive" -o /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef/zenodo_13755810_files_archive.zip
unzip -o /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef/zenodo_13755810_files_archive.zip -d /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef
```

Or:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/download_dispef.sh
```

## 3) Preprocess DISPEF-M (backbone-only)

This step:

- extracts only `N`, `CA`, `C` atoms
- stores residue identity + atom identity + coordinates
- keeps one chain/protein as one graph sample
- for DISPEF `.pt` mode, uses official `*_tr.pt` and `*_te.pt`, then samples validation from train only
- for structure-file mode, discovers official split if available

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m data.preprocess_dispef \
  --raw-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --val-fraction 0.1 \
  --seed 42
```

Auto-detect wrapper script:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/preprocess_dispef_m.sh
```

Notes:

- DISPEF Zenodo release is `.pt` based (`DISPEF_M_tr.pt`/`DISPEF_M_te.pt`); preprocessing now handles this directly.
- Residue identities are inferred directly from per-residue atom-name patterns in `.pt` mode.
- Optional: add `--fetch-uniprot-sequences` to override with UniProt sequence where available.
- Optional DSSP applies only to structure-file mode (`--input-format structure`).
- By default preprocessing clears old `processed/<dataset>/proteins/*.npz` before writing new output.

## 4) Generate Teacher Labels (cached)

Teacher outputs are cached once under:

- `/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher/dispef_m/*.npz`

Each cached file contains:

- `teacher_probs_residue`: `[L, 8]`
- `teacher_probs_node`: `[3L, 8]` (residue targets duplicated to `N/CA/C`)

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m teacher.generate_teacher_labels \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher \
  --provider esm3 \
  --esm-backend auto \
  --split all \
  --device cuda
```

Optional:

- add `--fetch-uniprot-sequences` if you want to override stored/inferred sequences
- add `--max-samples 10` for a quick auth/backend smoke test

If your ESM3 SDK/API differs, keep this abstraction and update `teacher/esm3_teacher.py` extraction logic.

Note: in `esm==3.2.x`, secondary-structure logits can be returned as 11-token vocab (`<pad>, <motif>, <unk>, G,H,I,T,E,B,S,C`). The wrapper projects these to SS8 (`G,H,I,T,E,B,S,C`) automatically.

### ESM3 access troubleshooting

If you see a 401/gated-model error for `EvolutionaryScale/esm3-sm-open-v1`, authenticate Hugging Face and keep cache on NVMe:

```bash
export HF_HOME=/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
huggingface-cli login
```

Then rerun teacher labeling with `--esm-backend local` (or `auto`).

If you use EvolutionaryScale Forge API instead of local HF weights:

```bash
export ESM_API_TOKEN=<your_forge_token>
python -m teacher.generate_teacher_labels ... --provider esm3 --esm-backend forge
```

### Use both GPUs for teacher labeling

Teacher labeling is one process per GPU. Use sharding to run two processes in parallel without overlap:

```bash
# Terminal 1 / tmux pane 1
python -m teacher.generate_teacher_labels \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher \
  --provider esm3 \
  --esm-backend local \
  --split all \
  --device cuda:0 \
  --num-shards 2 \
  --shard-id 0

# Terminal 2 / tmux pane 2
python -m teacher.generate_teacher_labels \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher \
  --provider esm3 \
  --esm-backend local \
  --split all \
  --device cuda:1 \
  --num-shards 2 \
  --shard-id 1
```

## 5) Train

Default config matches the paper baseline training schedule:

- `epochs=120`
- `batch_size=50`
- `Adam(lr=1e-3)`
- LR decay `gamma=0.9` every `3` epochs
- small weight decay

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m train.train \
  --config /opt/dlami/nvme/esm3_gnn_distill_baseline/configs/baseline_dispef_m.yaml \
  --run-name dispef_m_geo_gnn
```

Or:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/train_baseline.sh
```

Before training, ensure teacher cache coverage is complete. The trainer performs a preflight check and will stop with a detailed missing-label report if teacher files are incomplete.

### Optional Weights & Biases logging

Install once (if not already in env):

```bash
pip install wandb
```

Login and set API key in the shell:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
wandb login
```

Update config:

- `wandb.enabled: true`
- `wandb.entity: xingyuhu95-carnegie-mellon-university`
- `wandb.project: pLM_KD`
- `train.log_every_steps: 100` (batch-level W&B updates)

If W&B init fails (for example, missing API key), training now continues without W&B and logs a warning.

### Optional S3 autosync (checkpoints + logs)

The trainer can upload run artifacts to S3 during training.

Config (`configs/baseline_dispef_m.yaml`):

- `s3_sync.enabled: true`
- `s3_sync.bucket_prefix: s3://02750s3/active-learning-plm-distillation`
- `s3_sync.upload_last_each_epoch: true` (uploads `last.pt` every epoch)
- `s3_sync.upload_best: true`
- `s3_sync.upload_epoch_checkpoints: false` (optional)
- `s3_sync.upload_run_artifacts: true` (uploads `train.log` / `history.csv`)

Multiple experiments are isolated automatically under:

- `s3://.../<run_dir_name>/checkpoints/<run_dir_name>/...`
- `s3://.../<run_dir_name>/outputs/<run_dir_name>/...`

This avoids collisions across concurrent or repeated experiments.

Prerequisite: AWS credentials (instance role, `aws configure`, or SSO) must be available in the training shell.

### Run in background (survives SSH disconnect)

Recommended: `tmux`

```bash
tmux new -s distill_run
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill
cd /opt/dlami/nvme/esm3_gnn_distill_baseline
python -m train.train --config configs/baseline_dispef_m.yaml --run-name dispef_m_geo_gnn
# Detach: Ctrl+b then d
```

Reattach later:

```bash
tmux attach -t distill_run
```

Alternative: `nohup`

```bash
cd /opt/dlami/nvme/esm3_gnn_distill_baseline
nohup bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate /opt/dlami/nvme/envs/esm3_gnn_distill && python -m train.train --config configs/baseline_dispef_m.yaml --run-name dispef_m_geo_gnn' > /opt/dlami/nvme/esm3_gnn_distill_baseline/logs/train_nohup.log 2>&1 &
```

Watch progress:

```bash
tail -f /opt/dlami/nvme/esm3_gnn_distill_baseline/logs/train_nohup.log
```

## 6) Evaluate

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m eval.evaluate \
  --config /opt/dlami/nvme/esm3_gnn_distill_baseline/configs/baseline_dispef_m.yaml \
  --checkpoint /opt/dlami/nvme/esm3_gnn_distill_baseline/checkpoints/<RUN_DIR>/best.pt \
  --split test \
  --batch-size 16 \
  --save-predictions \
  --output-dir /opt/dlami/nvme/esm3_gnn_distill_baseline/outputs/eval/<RUN_DIR>
```

Reported metrics:

- teacher cross-entropy
- teacher KL divergence
- top-1 SS8 accuracy vs teacher argmax
- optional DSSP accuracy (if DSSP labels were produced)

## Distillation Losses Implemented

- hard-label CE: `cross_entropy(logits, class_index)`
- soft-label CE: `-sum(p_teacher * log_softmax(student_logits))`
- KL distillation: `KL(p_teacher || p_student)` with optional temperature

Total:

- `lambda_teacher * teacher_loss + lambda_dssp * dssp_loss`

## Active-Learning Readiness (without implementing AL yet)

- splits are explicit in `splits.json`: `train`, `val`, `test`, `pool_unassigned`
- teacher querying is isolated in `teacher/`
- `SplitIndex` helper supports moving IDs between splits incrementally

## What Matches the Paper vs Approximations

Closest matches:

- DISPEF-only pipeline (targeting DISPEF-M)
- backbone-only nodes (`N`, `CA`, `C`)
- amino acid + atom identity node features
- residue SS8 targets duplicated to the 3 backbone atoms
- softmax SS8 head with distillation training objective
- paper training schedule defaults (120 epochs, batch 50, Adam 1e-3, 0.9 decay every 3 epochs)

Approximations / documented deviations:

- student architecture is a compact PyG GNN baseline (`simple_gnn` or `geo_gnn`), not a full Schake reimplementation
- graph construction uses simple distance-cutoff edges with optional neighbor cap
- ESM3 wrapper is robust but may require minor API adaptation depending on installed ESM3 SDK version
- optional DSSP path uses Biopython DSSP, while Schake reference code also uses MDTraj-based utilities

## Reference

- https://github.com/ZhangGroup-MITChemistry/Schake_GNN/

This baseline intentionally stays limited to distillation-stage training/evaluation only.
