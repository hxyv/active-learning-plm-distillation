# ESM3 → GNN Distillation + Active Learning (DISPEF-M) — PSC

Reproduction of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

Running on **PSC Bridge-2**.

Implemented scope:

- `sequence + structure → teacher ESM3 SS8 probabilities → student GNN`
- student (Schake v2) trained on soft teacher targets plus DSSP auxiliary CE loss
- paper-faithful offline baseline: 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), patience=20
- **pool-based active learning loop** with passive (random), MC Dropout, EMC, and diversity strategies

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Project Layout

```
/ocean/projects/cis250233p/xhu15/
├── active-learning-plm-distillation/   ← this repo
│   ├── configs/
│   │   ├── psc_dispef_m.yaml           ← offline baseline config
│   │   └── al_psc_dispef_m.yaml        ← active learning config
│   ├── active_learning/
│   │   ├── pool_manager.py             ← pool/train split state
│   │   ├── acquisition.py              ← acquisition functions
│   │   └── al_loop.py                  ← AL loop CLI entry point
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── teacher/
│   ├── train/
│   ├── eval/
│   └── slurm/
│       ├── preprocess.slurm
│       ├── teacher.slurm
│       ├── train.slurm                 ← offline baseline training
│       ├── eval.slurm
│       └── al_loop.slurm              ← active learning loop
├── data/raw/dispef/                    ← DISPEF-M tensors
├── data/processed/                     ← preprocessed NPZ files
├── cache/teacher/                      ← ESM3 teacher label cache
├── checkpoints/                        ← saved model checkpoints
├── outputs/                            ← eval outputs + AL round results
└── logs/slurm/                         ← SLURM stdout/stderr
```

## 1 · One-time Setup

### 1.1 Directories

```bash
mkdir -p /ocean/projects/cis250233p/xhu15/{data/raw/dispef,data/processed,cache/teacher,checkpoints,logs/slurm,outputs,envs}
```

### 1.2 Clone the repo

```bash
cd /ocean/projects/cis250233p/xhu15
git clone -b psc-dev https://github.com/hxyue1/active-learning-plm-distillation.git
```

### 1.3 Create the conda environment

Run interactively on a login node (no GPU needed for env creation):

```bash
module load anaconda3/2024.10-1

conda create -p /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill python=3.10 -y
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

# PyTorch + CUDA (Bridge-2 nodes have CUDA 12.6 driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# PyG core
pip install torch-geometric

# torch-scatter and torch-cluster (must match torch + cuda build)
pip install torch-scatter torch-cluster \
  -f https://data.pyg.org/whl/torch-2.6.0+cu121.html

# ESM3 (EvolutionaryScale SDK — weights download automatically on first use via HuggingFace)
pip install esm httpx

# other dependencies
pip install mdtraj biopython pyyaml tqdm scikit-learn matplotlib requests wandb
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_scatter, torch_cluster; print('PyG extensions OK')"
```

## 2 · API Credentials

### Hugging Face (ESM3 local weights)

ESM3 is a gated model on HuggingFace. There is no separate download step — weights (~1.5 GB)
are downloaded automatically to `HF_HOME` the first time `slurm/teacher.slurm` runs.

Before that first run:

1. Request model access at `https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1`
2. Log in once on the login node:

```bash
module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

pip install huggingface_hub
huggingface-cli login        # paste your HF token when prompted
```

Token stored in `~/.cache/huggingface/token` and reused automatically by all subsequent jobs.

### Weights & Biases

```bash
module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill
wandb login                  # paste your W&B API key when prompted
```

Or non-interactively:

```bash
echo 'export WANDB_API_KEY="your_wandb_key"' >> ~/.bashrc
source ~/.bashrc
wandb login --relogin
```

W&B is **disabled by default** in the configs. Set `wandb.enabled: true` and `wandb.entity: xhu15` to enable.

## 3 · Download DISPEF

Download the full archive from Zenodo record 13755810:

```bash
RAW="/ocean/projects/cis250233p/xhu15/data/raw/dispef"

curl -L "https://zenodo.org/api/records/13755810/files-archive" \
  -o "${RAW}/dispef_archive.zip"

unzip "${RAW}/dispef_archive.zip" -d "${RAW}/"
rm "${RAW}/dispef_archive.zip"
```

The archive contains `DISPEF_M_tr.pt`, `DISPEF_M_te.pt`, and `dataset_prep.py` (plus other variants).

## 4 · Preprocess

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation
sbatch slurm/preprocess.slurm
```

For quick pipeline testing, edit `slurm/preprocess.slurm` and add `--max-files-per-split 200` to each preprocess command.

## 5 · Generate Teacher Labels

Teacher labels must cover **all** proteins (train + val + test + pool) before running either the
offline baseline or the active learning loop. The AL loop treats unlabeled proteins as unqueried,
but their cached labels are required upfront to simulate the oracle.

### Option A — Mock teacher (no ESM3, for pipeline testing)

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation

python scripts/generate_mock_teacher.py \
  --processed-root     /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name       dispef_m \
  --teacher-cache-root /ocean/projects/cis250233p/xhu15/cache/teacher \
  --label-smoothing    0.05
```

Produces smoothed one-hot labels from DSSP. Sufficient to verify the full pipeline end-to-end.

### Option B — Local ESM3 teacher (~16 GB VRAM)

ESM3 is run with backbone coordinates as structure input (in addition to sequence), so its SS8
predictions are conditioned on the same conformations used to compute DSSP labels.

```bash
# DISPEF-M (default)
sbatch slurm/teacher.slurm
```

To speed up labeling on a large dataset, submit multiple shards in parallel — each job processes
a non-overlapping subset:

```bash
sbatch --export=ALL,DATASET=dispef_m,NUM_SHARDS=4,SHARD_ID=0 slurm/teacher.slurm
sbatch --export=ALL,DATASET=dispef_m,NUM_SHARDS=4,SHARD_ID=1 slurm/teacher.slurm
sbatch --export=ALL,DATASET=dispef_m,NUM_SHARDS=4,SHARD_ID=2 slurm/teacher.slurm
sbatch --export=ALL,DATASET=dispef_m,NUM_SHARDS=4,SHARD_ID=3 slurm/teacher.slurm
```

Smoke test (run 10 samples to verify HF auth and model loading):

```bash
# edit teacher.slurm temporarily to add --max-samples 10
sbatch --export=ALL,DATASET=dispef_m slurm/teacher.slurm
```

Add `--overwrite` to regenerate existing cached labels.

## 6 · Offline Baseline Training

Trains the student on the **full** labeled training set (17,280 proteins).

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation
sbatch slurm/train.slurm
```

Config: [`configs/psc_dispef_m.yaml`](configs/psc_dispef_m.yaml)

Monitor:

```bash
squeue -u xhu15
tail -f /ocean/projects/cis250233p/xhu15/logs/slurm/<JOB_ID>.out
```

## 7 · Evaluate Offline Baseline

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation
sbatch --export=ALL,RUN_NAME=psc_schake_distill_<JOB_ID> slurm/eval.slurm
```

Results in `eval_summary_test.json`:

| Metric | Description |
|--------|-------------|
| `teacher_ce` | Soft-label cross-entropy vs teacher |
| `teacher_kl` | KL divergence vs teacher |
| `teacher_top1_acc` | Top-1 accuracy vs teacher argmax |
| `dssp_acc` | Accuracy vs DSSP hard labels |

## 8 · Active Learning Loop

The AL loop retrains the student from scratch each round on a growing labeled subset, using a
chosen acquisition strategy to select which proteins to query next.

### Setup

| Parameter | Value |
|-----------|-------|
| Candidate pool | first 12,000 training proteins |
| Initial labeled set | 2,500 (random) |
| Budget per round | 500 |
| Total rounds | 15 |
| Final labeled set | 10,000 |

Config: [`configs/al_psc_dispef_m.yaml`](configs/al_psc_dispef_m.yaml)

Per-round splits files and state are stored under the run's output directory; the original
`splits.json` is never modified, so the offline baseline can always be re-run.

### Passive baseline (random acquisition)

```bash
sbatch slurm/al_loop.slurm
# or explicitly:
sbatch --export=ALL,STRATEGY=random slurm/al_loop.slurm
```

### Resume an interrupted run

```bash
sbatch --export=ALL,STRATEGY=random,RESUME=1,RESUME_OUTPUT_DIR=/path/to/existing/run slurm/al_loop.slurm
```

### Output structure

```
outputs/al/<strategy>_<JOB_ID>/
├── al_state.json             ← pool/train state (updated each round)
├── al_results.json           ← aggregated per-round metrics for plotting
├── al_loop.log
├── splits/
│   ├── splits_round_00.json
│   ├── splits_round_01.json
│   └── ...
├── round_00/
│   ├── config_resolved.yaml
│   ├── train.log
│   ├── history.csv
│   ├── metrics_final.json
│   └── round_summary.json
└── round_01/
    └── ...
```

### Available strategies

| Strategy | Flag | Status | Description |
|----------|------|--------|-------------|
| Random (passive) | `random` | ✅ implemented | Uniform random sampling — establishes the floor |
| MC Dropout | `mc_dropout` | ✅ implemented | Mean predictive entropy over T=20 stochastic forward passes |
| EMC | `emc` | planned | Expected Model Change via output-layer gradient norms |
| Diversity | `diversity` | planned | Agglomerative clustering on node embeddings |

### MC Dropout details

The Schake v2 output MLP has no dropout by default. Setting `model.mc_dropout_p: 0.1` in the config (already set in `configs/al_psc_dispef_m.yaml`) prepends a `nn.Dropout` layer to the output network. Both training and acquisition use this layer, following Gal & Ghahramani (2016).

At acquisition time the model runs in eval mode with dropout layers kept active. Uncertainty per protein is the mean per-residue predictive entropy averaged over 20 passes:

$$U(p) = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{L_p} \sum_{r=1}^{L_p} H\!\left(\text{softmax}(\mathbf{z}_{t,r})\right)$$

To run MC Dropout acquisition:

```bash
sbatch --export=ALL,STRATEGY=mc_dropout slurm/al_loop.slurm
```

## Model Architecture

```
Input: pos [3L,3]  aa_idx [3L]  atom_idx [3L]   (3 backbone atoms per residue)
          │
  Dual-range radius graphs (on-the-fly):
    SAKE branch  :  all atoms,  0.25–1.0 nm  (local backbone geometry)
    SchNet branch:  CA only,    1.0–2.5 nm   (inter-residue contacts)
          │
  2 × (SAKE layer + SchNet layer)
          │
  3-layer MLP → 8 SS8 logits per node
```

## Distillation Loss

$$\mathcal{L} = \lambda_{\text{teacher}} \cdot \underbrace{-\sum_c p^{\text{teacher}}_c \log p^{\text{student}}_c}_{\text{soft CE}} + \lambda_{\text{DSSP}} \cdot \underbrace{-\log p^{\text{student}}_{y_{\text{DSSP}}}}_{\text{hard CE}}$$

Default: `lambda_teacher=1.0`, `lambda_dssp=1.0`.

## Reference

- Airas and Zhang (2026): https://arxiv.org/abs/2601.05388
- Schake GNN: https://github.com/ZhangGroup-MITChemistry/Schake_GNN/
- DISPEF dataset (Zenodo 13755810): https://zenodo.org/records/13755810
