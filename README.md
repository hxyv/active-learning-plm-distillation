# ESM3 → GNN Distillation Baseline (DISPEF-M) — Colab Demo

Reproduction of the distillation stage from:

- Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*

This branch (`colab-dev`) adapts the baseline for **Google Colab** exploration using a DISPEF-M subset, without requiring AWS or ESM3.  The full production pipeline lives on the `main` branch.

Implemented scope:

- `sequence + structure → teacher ESM3 SS8 probabilities → student GNN`
- student (Schake v2) trained on soft teacher targets plus DSSP auxiliary CE loss
- paper-faithful training: 120 epochs, batch 50, Adam lr=1e-3, StepLR(γ=0.9, step=3), validation split enabled (patience=20)

Not implemented:

- probability-to-energy conversion
- OpenMM / ML-MD / umbrella sampling / simulation workflows

## Quick Start (Colab)

Open `colab_demo.ipynb` in Google Colab (Runtime → T4 GPU) and run top to bottom.  The notebook covers:

1. Dependency installation
2. Download DISPEF-M from Zenodo (record 13755810, M subset only)
3. Preprocess a configurable subset (`SUBSET = 200` proteins by default)
4. Explore data structure — residue NPZ arrays, PyG graph object, Cα backbone visualisation
5. Explore model architecture — Schake v2 dual-range GNN, parameter count, forward pass
6. Loss functions — soft CE, KL-div, DSSP hard CE
7. Generate mock teacher labels (DSSP-based, no ESM3 needed)
8. Train the student GNN (10 epochs)
9. Evaluate and visualise results

## Project Layout

```text
active-learning-plm-distillation/
├── colab_demo.ipynb              ← Colab notebook (start here)
├── configs/
│   ├── colab_demo.yaml           ← Colab config (no S3/W&B, /content/ paths, 10 epochs)
│   └── paper_dispef_m.yaml       ← Full paper config (AWS, 120 epochs)
├── data/
│   ├── constants.py              ← AA / atom / SS8 vocabularies
│   ├── dssp.py                   ← DSSP label computation via mdtraj
│   ├── graph_builder.py          ← Residue NPZ → PyG Data (3L nodes)
│   ├── io_utils.py
│   ├── preprocess_dispef.py      ← PT tensors → per-protein NPZ + splits.json
│   └── pyg_dataset.py            ← DistillationGraphDataset + SplitIndex
├── models/
│   ├── factory.py
│   ├── gnn.py                    ← SimpleDistillGNN, GeoDistillGNN, SchakeDistillModel
│   └── vendor/
│       └── schake_model_v2.py    ← Official Schake v2 implementation
├── scripts/
│   ├── generate_mock_teacher.py  ← DSSP → mock teacher cache (Colab use)
│   ├── download_dispef.sh
│   ├── generate_teacher_labels.sh
│   ├── preprocess_dispef_m.sh
│   ├── train_baseline.sh
│   └── eval_baseline.sh
├── teacher/
│   ├── esm3_teacher.py
│   ├── generate_teacher_labels.py
│   └── label_cache.py
├── train/
│   ├── losses.py                 ← soft_ce, kl_div, hard_ce
│   ├── train.py
│   └── trainer.py
└── eval/
    ├── evaluate.py
    └── metrics.py
```

## Data Structure

### Residue-level NPZ (output of preprocessing)

Each protein is saved as `proteins/<sample_id>.npz`:

| Array | Shape | Notes |
|-------|-------|-------|
| `aa_idx` | `[L]` | Residue AA index 0–20 (20 standard AAs + X=unknown) |
| `coords` | `[L, 3, 3]` | Backbone N/CA/C coordinates in **nm** |
| `dssp_idx` | `[L]` | SS8 class 0–7: G H I T E B S C  (−100 = missing) |
| `sequence` | `[L]` | One-letter AA characters |

### Graph Data object (input to GNN)

`graph_builder.py` expands residue arrays to **per-node** arrays by repeating each residue 3× (one entry per backbone atom N, CA, C).  A protein with **L** residues becomes a graph with **3L nodes**.

| Tensor | Shape | Notes |
|--------|-------|-------|
| `pos` | `[3L, 3]` | Coordinates in nm, zero-centred |
| `aa_idx` | `[3L]` | Residue AA index, repeated 3× |
| `atom_idx` | `[3L]` | Atom type 0=N 1=CA 2=C, tiled per residue |
| `dssp_idx` | `[3L]` | SS8 label, repeated 3× |
| `node_to_residue` | `[3L]` | Maps node index → residue index |
| `teacher_probs` | `[3L, 8]` | Soft SS8 distribution from ESM3 (if cached) |

`edge_index` / `edge_attr` are empty when `cutoff=0.001` — Schake v2 builds its own radius graphs internally each forward pass.

## Model Architecture — Schake v2

```
Input: pos [3L,3]  aa_idx [3L]  atom_idx [3L]
          │
  Index remapping → Schake vocab
          │
  Dual-range radius graphs (built on-the-fly):
    SAKE branch  :  all atoms,  0.25 – 1.0 nm  (local / covalent geometry)
    SchNet branch:  CA only,    1.0  – 2.5 nm  (inter-residue contacts)
          │
  2 × (SAKE layer + SchNet layer + LayerNorm)
          │
  3-layer MLP output head → 8 SS8 logits per node
```

Key config (`configs/colab_demo.yaml` / `configs/paper_dispef_m.yaml`):

| Parameter | Value |
|-----------|-------|
| `hidden_channels` | 32 |
| `num_layers` | 2 |
| `kernel_size` | 18 |
| `sake_low/high_cut` | 0.25 / 1.0 nm |
| `schnet_low/high_cut` | 1.0 / 2.5 nm |
| `num_heads` | 4 |
| `num_out_layers` | 3 |

## Distillation Losses

$$\mathcal{L} = \lambda_{\text{teacher}} \cdot \underbrace{-\sum_c p^{\text{teacher}}_c \log p^{\text{student}}_c}_{\text{soft CE}} + \lambda_{\text{DSSP}} \cdot \underbrace{-\log p^{\text{student}}_{y_{\text{DSSP}}}}_{\text{hard CE}}$$

- **Soft CE** (`lambda_teacher=1.0`): match the full ESM3 SS8 probability distribution
- **DSSP CE** (`lambda_dssp=1.0`): auxiliary hard-label supervision from mdtraj
- **KL divergence**: alternative to soft CE (`teacher_loss_type: kl`)

## Colab Pipeline (step by step)

### 1 · Install dependencies

```python
import torch
torch_ver = torch.__version__.split('+')[0]
cuda_ver  = torch.version.cuda.replace('.', '')
pyg_url   = f'https://data.pyg.org/whl/torch-{torch_ver}+cu{cuda_ver}.html'

!pip install -q torch-geometric
!pip install -q torch-scatter torch-cluster -f {pyg_url}
!pip install -q biopython 'mdtraj==1.10.1' pyyaml tqdm scikit-learn matplotlib
```

`torch-scatter` must match the installed PyTorch + CUDA build — the detection above handles this automatically.

### 2 · Clone and configure workspace

```python
!git clone -b colab-dev https://github.com/hxyue1/active-learning-plm-distillation.git
import os, sys, yaml
os.chdir('/content/active-learning-plm-distillation')
sys.path.insert(0, os.getcwd())

WORKSPACE = '/content/dispef_ws'
SUBSET    = 200   # proteins to preprocess; set 0 for full DISPEF-M

# Write runtime config with correct workspace paths
cfg = yaml.safe_load(open('configs/colab_demo.yaml'))
cfg['paths'].update({k: f'{WORKSPACE}/{v}' for k, v in {
    'raw_dispef_root'   : 'data/raw/dispef',
    'processed_root'    : 'data/processed',
    'teacher_cache_root': 'cache/teacher',
    'outputs_root'      : 'outputs',
    'checkpoints_root'  : 'checkpoints',
}.items()})
with open('/tmp/colab_runtime.yaml', 'w') as f:
    yaml.dump(cfg, f)
```

### 3 · Download DISPEF-M

Zenodo record 13755810 is the M subset only — not the full DISPEF dataset.

```python
!curl -L 'https://zenodo.org/api/records/13755810/files-archive' \
      -o {WORKSPACE}/data/raw/dispef/archive.zip
!unzip -q {WORKSPACE}/data/raw/dispef/archive.zip \
       -d {WORKSPACE}/data/raw/dispef/
```

### 4 · Preprocess

```bash
python -m data.preprocess_dispef \
  --raw-root       /content/dispef_ws/data/raw/dispef  \
  --processed-root /content/dispef_ws/data/processed   \
  --dataset-name   dispef_m                            \
  --val-fraction 0.1 --seed 42                         \
  --max-files 200
```

`--max-files N` caps total proteins (train + test combined). Omit for full dataset.

### 5 · Generate mock teacher labels (no ESM3 needed)

```bash
python scripts/generate_mock_teacher.py \
  --processed-root     /content/dispef_ws/data/processed \
  --dataset-name       dispef_m                          \
  --teacher-cache-root /content/dispef_ws/cache/teacher  \
  --label-smoothing 0.05
```

Creates one-hot (+ smoothed) teacher probability files from DSSP labels.  With mock labels the soft-CE loss is equivalent to hard-label CE — sufficient for pipeline testing.  Real distillation requires ESM3 labels (see below).

### 6 · Train

```bash
python -m train.train \
  --config   /tmp/colab_runtime.yaml \
  --run-name colab_demo
```

### 7 · Evaluate

```bash
python -m eval.evaluate \
  --config     /tmp/colab_runtime.yaml              \
  --checkpoint /content/dispef_ws/checkpoints/<RUN>/best.pt \
  --split test                                      \
  --output-dir /content/dispef_ws/outputs/eval/<RUN>
```

Metrics saved to `eval_summary_test.json`:

- `teacher_ce` — soft-label cross-entropy
- `teacher_kl` — KL divergence
- `teacher_top1_acc` — top-1 accuracy vs teacher argmax
- `dssp_acc` — accuracy vs DSSP hard labels

## Real Distillation (with ESM3 teacher)

To generate actual ESM3 SS8 teacher labels (requires large GPU or Forge API key):

```bash
# Local ESM3 (needs ~16 GB VRAM)
python -m teacher.generate_teacher_labels \
  --processed-root /content/dispef_ws/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /content/dispef_ws/cache/teacher \
  --provider esm3 --esm-backend local --split all --device cuda

# EvolutionaryScale Forge API (no local GPU needed)
export ESM_API_TOKEN=<your_forge_token>
python -m teacher.generate_teacher_labels \
  --processed-root /content/dispef_ws/data/processed \
  --dataset-name dispef_m \
  --teacher-cache-root /content/dispef_ws/cache/teacher \
  --provider esm3 --esm-backend forge --split all
```

Note: ESM3 returns SS8 logits as an 11-token vocab in `esm==3.2.x` — the wrapper projects them to the 8-class SS8 order `[G,H,I,T,E,B,S,C]` automatically.

## Active-Learning Readiness

- splits stored explicitly in `splits.json`: `train`, `val`, `test`, `pool_unassigned`
- `SplitIndex` in `data/pyg_dataset.py` supports moving IDs between splits for future AL cycles
- teacher querying isolated in `teacher/`

## Reference

- Airas and Zhang (2026): https://arxiv.org/abs/2601.05388
- Schake GNN source: https://github.com/ZhangGroup-MITChemistry/Schake_GNN/
