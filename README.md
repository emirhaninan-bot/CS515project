> CS515 — Deep Learning, Spring 2026 · Emirhan İnan · Burak Güler


# lncRNA SNP Pathogenicity Prediction with a Graph Attention Network

Classifying the effect of single-nucleotide polymorphisms (SNPs) on long non-coding RNAs
(lncRNAs) through graph-based deep learning.

## Repository structure

```
CS515project/
├── main.py                       # CLI entry point — dispatches train / fusion / test
├── parameters.py                 # All hyperparameters, paths, and dataset args (argparse)
├── train.py                      # Training loops + FocalLoss + gene-level split
├── test.py                       # Held-out evaluation + "hard case" subset analysis
├── visualize.py                  # Embedding / probability visualisations (t-SNE, histograms)
├── models/
│   ├── GAT_model.py
│   │   ├── GATBlock              
│   │   ├── PerturbationGAT       
│   │   └── LateFusionGNN         
│
├── data/
├── checkpoints/

```



---

## Usage

Training and evaluation are driven through `main.py` with three modes.

```bash
# 1. Train the PerturbationGAT 
python main.py train

# 2. Train the late-fusion head
python main.py fusion

# 3. Evaluate the late-fusion model on the held-out gene-level test split
python main.py test
```

All hyperparameters and paths are configurable via flags (see `parameters.py`), e.g.:

```bash
python main.py train --batch-size 32 --phase2-epochs 50 --hidden-dim 256
python main.py fusion --fusion-epochs 20 --data-dir /path/to/data
```

Generate embedding/probability visualisations:

```bash
python visualize.py --model fusion --split test          # t-SNE + probability histogram
python visualize.py --model gat --color label            # structural-only embeddings
```

Outputs are written to `Visualization outputs/`.

---

## Requirements

- Python 3.12
- PyTorch
- PyTorch Geometric (`torch_geometric` — `GATv2Conv`, `GlobalAttention`, `global_max_pool`)
- NumPy, SciPy, scikit-learn (metrics, PCA, t-SNE)
- Matplotlib

---

