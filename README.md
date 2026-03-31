# ESM3 → GNN Distillation Baseline (DISPEF-M) — PSC

Reproduction of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

Running on **PSC Bridge-2** (H100 GPU, `GPU-shared` partition, allocation `cis250233p`).

Implemented scope:

- `sequence + structure → teacher ESM3 SS8 probabilities → student GNN`
- student (Schake v2) trained on soft teacher targets plus DSSP auxiliary CE loss
- paper-faithful training: 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), patience=20

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Project Layout

```
active-learning-plm-distillation/
├── configs/
│   ├── psc_dispef_m.yaml         ← PSC config (no S3/W&B, ocean paths)
│   └── paper_dispef_m.yaml       ← original AWS config
├── data/
│   ├── constants.py
│   ├── dssp.py
│   ├── graph_builder.py
│   ├── io_utils.py
│   ├── preprocess_dispef.py
│   └── pyg_dataset.py
├── models/
│   ├── factory.py
│   ├── gnn.py
│   └── vendor/schake_model_v2.py
├── scripts/
│   ├── generate_mock_teacher.py
│   ├── generate_teacher_labels.sh
│   ├── preprocess_dispef_m.sh
│   ├── train_baseline.sh
│   └── eval_baseline.sh
├── teacher/
│   ├── esm3_teacher.py
│   ├── generate_teacher_labels.py
│   └── label_cache.py
├── train/
│   ├── losses.py
│   ├── train.py
│   └── trainer.py
├── eval/
│   ├── evaluate.py
│   └── metrics.py
└── psc.slurm                     ← SLURM job script
```

## 1 · One-time Setup

### 1.1 Workspace

```bash
WORKSPACE="/ocean/projects/cis250233p/${USER}/esm3_gnn_distill"
mkdir -p "${WORKSPACE}"/{data/raw/dispef,data/processed,cache/teacher,checkpoints,logs/slurm,outputs}
```

### 1.2 Clone the repo

```bash
cd "${WORKSPACE}"
git clone -b psc-dev https://github.com/hxyue1/active-learning-plm-distillation.git
```

### 1.3 Create the conda environment

Run interactively on a login node (no GPU needed for env creation):

```bash
module load anaconda3/2024.10-1

conda create -n idl_diffusion_env python=3.10 -y
conda activate idl_diffusion_env

# PyTorch 2.3 + CUDA 12.1 (matches Bridge-2 H100 drivers)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# PyG core
pip install torch-geometric

# torch-scatter and torch-cluster (must match torch + cuda build)
pip install torch-scatter torch-cluster \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# other dependencies
pip install mdtraj biopython pyyaml tqdm scikit-learn matplotlib requests wandb
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_scatter, torch_cluster; print('PyG extensions OK')"
```

## 2 · Download DISPEF-M

```bash
WORKSPACE="/ocean/projects/cis250233p/${USER}/esm3_gnn_distill"
RAW="${WORKSPACE}/data/raw/dispef"

# DISPEF-M train/test tensors + dataset helper (~3 GB total)
wget -O "${RAW}/DISPEF_M_tr.pt" \
  "https://zenodo.org/records/13755810/files/DISPEF_M_tr.pt"
wget -O "${RAW}/DISPEF_M_te.pt" \
  "https://zenodo.org/records/13755810/files/DISPEF_M_te.pt"
wget -O "${RAW}/dataset_prep.py" \
  "https://zenodo.org/records/13755810/files/dataset_prep.py"
```

## 3 · Preprocess

```bash
WORKSPACE="/ocean/projects/cis250233p/${USER}/esm3_gnn_distill"
cd "${WORKSPACE}/active-learning-plm-distillation"

module load anaconda3/2024.10-1
conda activate idl_diffusion_env

# Full dataset
python -m data.preprocess_dispef \
  --raw-root       "${WORKSPACE}/data/raw/dispef" \
  --processed-root "${WORKSPACE}/data/processed" \
  --dataset-name   dispef_m \
  --val-fraction   0.1 \
  --seed           42

# Or cap per split for quick testing
python -m data.preprocess_dispef \
  --raw-root       "${WORKSPACE}/data/raw/dispef" \
  --processed-root "${WORKSPACE}/data/processed" \
  --dataset-name   dispef_m \
  --val-fraction   0.1 \
  --seed           42 \
  --max-files-per-split 200
```

`--max-files-per-split N` caps train and test independently so both splits always have data.

## 4 · Generate Teacher Labels

### Option A — Mock teacher (no ESM3, for pipeline testing)

```bash
python scripts/generate_mock_teacher.py \
  --processed-root     "${WORKSPACE}/data/processed" \
  --dataset-name       dispef_m \
  --teacher-cache-root "${WORKSPACE}/cache/teacher" \
  --label-smoothing    0.05
```

Produces smoothed one-hot labels from DSSP. Sufficient to verify the full pipeline end-to-end.

### Option B — Real ESM3 teacher (~16 GB VRAM)

```bash
# Local ESM3
python -m teacher.generate_teacher_labels \
  --processed-root     "${WORKSPACE}/data/processed" \
  --dataset-name       dispef_m \
  --teacher-cache-root "${WORKSPACE}/cache/teacher" \
  --provider esm3 --esm-backend local --split all --device cuda

# EvolutionaryScale Forge API (no local GPU needed)
export ESM_API_TOKEN=<your_forge_token>
python -m teacher.generate_teacher_labels \
  --processed-root     "${WORKSPACE}/data/processed" \
  --dataset-name       dispef_m \
  --teacher-cache-root "${WORKSPACE}/cache/teacher" \
  --provider esm3 --esm-backend forge --split all
```

## 5 · Submit Training Job

```bash
cd "${WORKSPACE}/active-learning-plm-distillation"
sbatch psc.slurm
```

Monitor:

```bash
squeue -u ${USER}
tail -f /ocean/projects/cis250233p/${USER}/esm3_gnn_distill/logs/slurm/<JOB_ID>.out
```

Training config: [`configs/psc_dispef_m.yaml`](configs/psc_dispef_m.yaml)  
W&B is disabled by default; set `wandb.enabled: true` and fill in `wandb.entity` to enable.

## 6 · Evaluate

```bash
WORKSPACE="/ocean/projects/cis250233p/${USER}/esm3_gnn_distill"
RUN="psc_schake_distill_<JOB_ID>"

python -m eval.evaluate \
  --config     "${WORKSPACE}/active-learning-plm-distillation/configs/psc_dispef_m.yaml" \
  --checkpoint "${WORKSPACE}/checkpoints/${RUN}/best.pt" \
  --split      test \
  --output-dir "${WORKSPACE}/outputs/eval/${RUN}"
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
