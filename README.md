## Introduction
Protein language model distillation to graph neural networks is often data-intensive, requiring large labeled protein datasets for supervised training.

We propose a pool-based active learning framework to improve sample efficiency in distilling protein language models into graph neural networks for secondary structure (SS8) prediction.

This project reproduces the distillation pipeline from:
Airas and Zhang (2026), *Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model*


## Method
We integrate EMC-based and clustering-based active learning strategies into an existing teacher-student distillation pipeline for SS8 prediction.

The framework consists of:
- Teacher: ESM3 model generating SS8 probability distributions
- Student: GNN (Schake v2) trained on soft teacher targets + DSSP cross-entropy loss
- Input: protein sequence + structural features

We further introduce a pool-based active learning loop with two acquisition strategies:
- Expected Model Change (EMC)-based uncertainty sampling
- Hierarchical clustering-based diversity sampling

## Key Contributions
- Implemented EMC-based acquisition function for active learning in protein representation learning
- Designed hierarchical clustering-based diversity sampling for efficient protein subset selection
- Integrated both acquisition strategies into an existing distillation pipeline with iterative training


## Results
- Achieved 91.67% SS8 prediction accuracy
- Reduced labeled data requirement by ~25% compared to full supervision

## Reference

- Airas and Zhang (2026). Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model. https://arxiv.org/abs/2601.05388
- Schake GNN implementation: https://github.com/ZhangGroup-MITChemistry/Schake_GNN
- DISPEF dataset: https://zenodo.org/records/13755810