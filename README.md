# ESM3 -> GNN Distillation Baseline (DISPEF-M)

Reproduction of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

Implemented scope:

- `sequence + structure -> teacher ESM3 SS8 probabilities -> student GNN`
- student (Schake v2) trained on soft teacher targets plus DSSP auxiliary CE loss
- paper-faithful training: 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), no validation split

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Project Layout

```text
/opt/dlami/nvme/esm3_gnn_distill_baseline
├── cache/
├── checkpoints/
├── configs/
│   └── paper_dispef_m.yaml
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
│   ├── gnn.py
│   └── vendor/
│       └── schake_model_v2.py
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
pip install torch-geometric
pip install mdtraj==1.10.1
```

Single-command alternative:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/setup_env_nvme.sh
```

## Startup API Checklist

Set credentials in every new shell (or load them from a local, untracked env file).

```bash
# Weights & Biases (training monitoring)
export WANDB_API_KEY=<your_wandb_api_key>

# Hugging Face (required for local ESM3 gated model access)
export HF_HOME=/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# optional explicit token:
# export HUGGINGFACE_HUB_TOKEN=<your_hf_token>

# EvolutionaryScale Forge API (only if using --esm-backend forge)
# export ESM_API_TOKEN=<your_forge_token>

# AWS/S3 (required if s3_sync.enabled=true or manual aws s3 sync)
# Prefer instance role. If not using instance role, export:
# export AWS_ACCESS_KEY_ID=<...>
# export AWS_SECRET_ACCESS_KEY=<...>
# export AWS_SESSION_TOKEN=<...>   # if temporary credentials
# export AWS_DEFAULT_REGION=us-east-1
```

Notes:

- Keep tokens out of git. Do not commit them to configs or scripts.
- For HF login flow, run `huggingface-cli login` once in the active environment.

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

## 3) Preprocess DISPEF-M

This step:

- parses `DISPEF_M_tr.pt` and `DISPEF_M_te.pt` directly
- extracts `N`, `CA`, `C`, `O` backbone coordinates (in nanometers)
- computes 8-class DSSP labels from backbone coordinates using mdtraj (version 1.10.1), matching the paper
- stores residue identity + atom identity + coordinates + DSSP labels per protein as NPZ
- applies official train/test split; no validation holdout (paper protocol)

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m data.preprocess_dispef \
  --raw-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/raw/dispef \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --seed 42
```

Auto-detect wrapper script:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/preprocess_dispef_m.sh
```

Notes:

- DISPEF coordinates are stored in nanometers; mdtraj accepts nm natively — no unit conversion needed.
- DSSP labels are class indices 0-7 mapping to `["G","H","I","T","E","B","S","C"]`.
- Preprocessing clears old `processed/<dataset>/proteins/*.npz` before writing new output.

## 4) Generate Teacher Labels (cached)

Teacher outputs are cached once under:

- `/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher/dispef_m/*.npz`

Each cached file contains:

- `teacher_probs_residue`: `[L, 8]`
- `teacher_probs_node`: `[3L, 8]` (residue targets duplicated to `N/CA/C`)

ESM3 is run with the DISPEF-M backbone coordinates as structure input (in addition to sequence), so its SS8 predictions are conditioned on the same conformations used to compute mdtraj DSSP labels. This keeps the two training signals consistent and avoids gradient conflict during distillation.

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

python -m teacher.generate_teacher_labels \
  --processed-root /opt/dlami/nvme/esm3_gnn_distill_baseline/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /opt/dlami/nvme/esm3_gnn_distill_baseline/cache/teacher \
  --provider esm3 \
  --esm-backend local \
  --split all \
  --device cuda
```

To regenerate existing labels with structure conditioning, add `--overwrite`.

Optional:

- add `--fetch-uniprot-sequences` if you want to override stored/inferred sequences
- add `--max-samples 10` for a quick auth/backend smoke test

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

Training uses the Schake v2 architecture with the paper protocol:

- Schake v2 (`model.name: schake`, `hidden_channels=32`, `num_layers=2`)
- official DISPEF-M train/test split; no validation holdout
- 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), weight decay 1e-6
- loss: teacher soft-CE + DSSP auxiliary CE (`lambda_teacher=1.0`, `lambda_dssp=1.0`)
- Schake builds its own radius graph internally (SAKE: 0.25–1.0 nm, SchNet: 1.0–2.5 nm)
- mixed precision (AMP) enabled

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill

cd /opt/dlami/nvme/esm3_gnn_distill_baseline
python -m train.train \
  --config configs/paper_dispef_m.yaml \
  --run-name baseline
```

Or:

```bash
bash /opt/dlami/nvme/esm3_gnn_distill_baseline/scripts/train_baseline.sh
```

Expected throughput: ~70 seconds/epoch → ~2.3 hours for 120 epochs on a single A10/A100 GPU.

The trainer performs a preflight check and will stop with a detailed missing-label report if teacher cache files are incomplete.

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

Config (`configs/paper_dispef_m.yaml`):

- `wandb.enabled: true`
- `wandb.entity: xingyuhu95-carnegie-mellon-university`
- `wandb.project: pLM_KD`
- `wandb.name: baseline`
- `train.log_every_steps: 100` (batch-level W&B updates)

If W&B init fails (for example, missing API key), training continues without W&B and logs a warning.

### Optional S3 autosync (checkpoints + logs)

The trainer can upload run artifacts to S3 during training.

Config (`configs/paper_dispef_m.yaml`):

- `s3_sync.enabled: true`
- `s3_sync.bucket_prefix: s3://02750s3/active-learning-plm-distillation`
- `s3_sync.upload_last_each_epoch: true` (uploads `last.pt` every epoch)
- `s3_sync.upload_best: true`
- `s3_sync.upload_epoch_checkpoints: false` (optional)
- `s3_sync.upload_run_artifacts: true` (uploads `train.log` / `history.csv`)

Multiple experiments are isolated automatically under:

- `s3://.../<run_dir_name>/checkpoints/<run_dir_name>/...`
- `s3://.../<run_dir_name>/outputs/<run_dir_name>/...`

Prerequisite: AWS credentials (instance role, `aws configure`, or SSO) must be available in the training shell.

### Instance stop/start recovery notes

If your NVMe data is ephemeral, do this before stopping the instance:

1. Push code changes to GitHub.
2. Sync critical artifacts to S3:
   - `checkpoints/`
   - `outputs/`
   - `cache/teacher/`
   - `data/processed/`
3. Optionally skip `data/raw/` if you can re-download DISPEF.

Example:

```bash
export BUCKET_PREFIX="s3://02750s3/active-learning-plm-distillation/$(date -u +%Y%m%d_%H%M%S)"
cd /opt/dlami/nvme/esm3_gnn_distill_baseline
aws s3 sync checkpoints "${BUCKET_PREFIX}/checkpoints/"
aws s3 sync outputs "${BUCKET_PREFIX}/outputs/"
aws s3 sync cache/teacher "${BUCKET_PREFIX}/teacher_cache/"
aws s3 sync data/processed "${BUCKET_PREFIX}/data_processed/"
```

After starting a new instance:

1. Clone/pull repo.
2. Recreate/activate conda env.
3. Restore artifacts from S3.
4. Resume training from `checkpoints/<RUN_DIR>/last.pt` if needed.

### Run in background (survives SSH disconnect)

Recommended: `tmux`

```bash
tmux new -s distill_run
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/envs/esm3_gnn_distill
cd /opt/dlami/nvme/esm3_gnn_distill_baseline
python -m train.train --config configs/paper_dispef_m.yaml --run-name baseline
# Detach: Ctrl+b then d
```

Reattach later:

```bash
tmux attach -t distill_run
```

Alternative: `nohup`

```bash
cd /opt/dlami/nvme/esm3_gnn_distill_baseline
nohup /opt/dlami/nvme/envs/esm3_gnn_distill/bin/python -m train.train \
  --config configs/paper_dispef_m.yaml \
  --run-name baseline \
  > logs/train_nohup.log 2>&1 &
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
  --config /opt/dlami/nvme/esm3_gnn_distill_baseline/configs/paper_dispef_m.yaml \
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
- DSSP accuracy (hard labels from mdtraj)

## Distillation Losses

- soft-label CE: `-sum(p_teacher * log_softmax(student_logits))`
- DSSP auxiliary CE: `cross_entropy(logits, dssp_class_index, ignore_index=-100)`
- KL distillation: `KL(p_teacher || p_student)` with optional temperature

Total:

- `lambda_teacher * teacher_loss + lambda_dssp * dssp_loss`

## Active-Learning Readiness (without implementing AL yet)

- splits are explicit in `splits.json`: `train`, `val`, `test`, `pool_unassigned`
- teacher querying is isolated in `teacher/`
- `SplitIndex` helper supports moving IDs between splits incrementally


## Reference

- Airas and Zhang (2026): https://arxiv.org/abs/2601.05388
- Schake GNN source: https://github.com/ZhangGroup-MITChemistry/Schake_GNN/
