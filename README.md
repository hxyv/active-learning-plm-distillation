# ESM3 → GNN Distillation Baseline (DISPEF-M) — PSC

Reproduction of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

Running on **PSC Bridge-2**.

Implemented scope:

- `sequence + structure → teacher ESM3 SS8 probabilities → student GNN`
- student (Schake v2) trained on soft teacher targets plus DSSP auxiliary CE loss
- paper-faithful training: 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), patience=20

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Project Layout

```
/ocean/projects/cis250233p/xhu15/
├── active-learning-plm-distillation/   ← this repo
│   ├── configs/psc_dispef_m.yaml
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── teacher/
│   ├── train/
│   ├── eval/
│   └── psc.slurm
├── data/raw/dispef/                    ← DISPEF-M tensors
├── data/processed/                     ← preprocessed NPZ files
├── cache/teacher/                      ← ESM3 teacher label cache
├── checkpoints/                        ← saved model checkpoints
├── outputs/                            ← eval outputs
└── logs/slurm/                         ← SLURM stdout/stderr
```

## 1 · One-time Setup

### 1.1 Directories

```bash
mkdir -p /ocean/projects/cis250233p/xhu15/{data/raw/dispef,data/processed,cache/teacher,checkpoints,logs/slurm,outputs}
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
module load cuda/12.4.0

conda create -n esm3_gnn_distill python=3.10 -y
source activate esm3_gnn_distill

# PyTorch 2.4 + CUDA 12.4 (matches Bridge-2 cuda/12.4.0)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# PyG core
pip install torch-geometric

# torch-scatter and torch-cluster (must match torch + cuda build)
pip install torch-scatter torch-cluster \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

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

ESM3 is a gated model — request access at
`https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1` before downloading weights.

```bash
module load anaconda3/2024.10-1
source activate esm3_gnn_distill

pip install huggingface_hub
huggingface-cli login        # paste your HF token when prompted
```

Or non-interactively (for use inside SLURM jobs):

```bash
export HF_TOKEN="hf_..."
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
```

Token stored in `~/.cache/huggingface/token` and reused automatically.  
Skip if using the Forge API backend instead.

### EvolutionaryScale Forge API (ESM3 via API, no local weights needed)

Get a token at `https://forge.evolutionaryscale.ai`.

```bash
# add to ~/.bashrc to persist across sessions
echo 'export ESM_API_TOKEN="your_forge_token"' >> ~/.bashrc
source ~/.bashrc
```

### Weights & Biases

```bash
module load anaconda3/2024.10-1
source activate esm3_gnn_distill
wandb login                  # paste your W&B API key when prompted
```

Or non-interactively:

```bash
echo 'export WANDB_API_KEY="your_wandb_key"' >> ~/.bashrc
source ~/.bashrc
wandb login --relogin
```

W&B is **disabled by default** in [`configs/psc_dispef_m.yaml`](configs/psc_dispef_m.yaml).  
Set `wandb.enabled: true` and `wandb.entity: xhu15` to enable it.

## 3 · Download DISPEF

Download the full archive from Zenodo record 13755810:

```bash
RAW="/ocean/projects/cis250233p/xhu15/data/raw/dispef"

curl -L "https://zenodo.org/api/records/13755810/files-archive" \
  -o "${RAW}/dispef_archive.zip"

unzip "${RAW}/dispef_archive.zip" -d "${RAW}/"
rm "${RAW}/dispef_archive.zip"
```

The archive contains `DISPEF_M_tr.pt`, `DISPEF_M_te.pt`, `DISPEF_S_tr.pt`, `DISPEF_S_te.pt`, and `dataset_prep.py`.

## 4 · Preprocess

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation
module load anaconda3/2024.10-1
source activate esm3_gnn_distill

# DISPEF-M
python -m data.preprocess_dispef \
  --raw-root       /ocean/projects/cis250233p/xhu15/data/raw/dispef \
  --processed-root /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name   dispef_m \
  --val-fraction   0.1 \
  --seed           42

# DISPEF-S
python -m data.preprocess_dispef \
  --raw-root       /ocean/projects/cis250233p/xhu15/data/raw/dispef \
  --processed-root /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name   dispef_s \
  --val-fraction   0.1 \
  --seed           42
```

For quick pipeline testing, add `--max-files-per-split 200` to cap train and test independently.

## 5 · Generate Teacher Labels

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

### Option B — Real ESM3 teacher (~16 GB VRAM)

```bash
# Local ESM3 (submit as GPU job or run interactively on GPU node)
python -m teacher.generate_teacher_labels \
  --processed-root     /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name       dispef_m \
  --teacher-cache-root /ocean/projects/cis250233p/xhu15/cache/teacher \
  --provider esm3 --esm-backend local --split all --device cuda

# EvolutionaryScale Forge API (no local GPU needed)
python -m teacher.generate_teacher_labels \
  --processed-root     /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name       dispef_m \
  --teacher-cache-root /ocean/projects/cis250233p/xhu15/cache/teacher \
  --provider esm3 --esm-backend forge --split all
```

## 6 · Submit Training Job

```bash
cd /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation
sbatch psc.slurm
```

Monitor:

```bash
squeue -u xhu15
tail -f /ocean/projects/cis250233p/xhu15/logs/slurm/<JOB_ID>.out
```

Training config: [`configs/psc_dispef_m.yaml`](configs/psc_dispef_m.yaml)  
W&B is disabled by default; set `wandb.enabled: true` and `wandb.entity: xhu15` to enable.

## 7 · Evaluate

```bash
RUN="psc_schake_distill_<JOB_ID>"

python -m eval.evaluate \
  --config     /ocean/projects/cis250233p/xhu15/active-learning-plm-distillation/configs/psc_dispef_m.yaml \
  --checkpoint /ocean/projects/cis250233p/xhu15/checkpoints/${RUN}/best.pt \
  --split      test \
  --output-dir /ocean/projects/cis250233p/xhu15/outputs/eval/${RUN}
```

Results in `eval_summary_test.json`:

| Metric | Description |
|--------|-------------|
| `teacher_ce` | Soft-label cross-entropy vs teacher |
| `teacher_kl` | KL divergence vs teacher |
| `teacher_top1_acc` | Top-1 accuracy vs teacher argmax |
| `dssp_acc` | Accuracy vs DSSP hard labels |

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
