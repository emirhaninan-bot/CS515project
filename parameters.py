import argparse
import os


def get_args():
    """Parse and return all command-line arguments with sensible defaults.

    All hyperparameters, paths, and dataset settings are centralised here so
    that every run can be fully reproduced or varied without touching source
    files. Derived checkpoint paths are appended to the returned Namespace
    after parsing, built from --checkpoint-dir.

    Returns:
        argparse.Namespace: Parsed arguments including derived checkpoint paths
            (expert_weights, fusion_weights, best_gat_weights).
    """
    parser = argparse.ArgumentParser(
        description="lncRNA Variant Pathogenicity Prediction with Siamese GAT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "mode", nargs="?", default="train",
        choices=["train", "fusion", "test"],
        help="train=PerturbationGAT, fusion=LateFusion head, test=evaluate",
    )

    # ── Directories ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir", default="data", metavar="DIR",
        help="Directory containing FASTAs, CSVs, and *_matrices/ folders",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", metavar="DIR",
        help="Directory for saving and loading model weights",
    )

    # ── PerturbationGAT training ──────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="Batch size for PerturbationGAT training")
    parser.add_argument("--phase1-epochs", type=int, default=5, metavar="N",
                        help="Phase 1 epochs — feature branch only, GNN frozen")
    parser.add_argument("--phase2-epochs", type=int, default=25, metavar="N",
                        help="Phase 2 epochs — full model, GNN unfrozen")
    parser.add_argument("--phase1-lr", type=float, default=5e-3, metavar="LR",
                        help="Learning rate for Phase 1 (feature branch warmup)")
    parser.add_argument("--phase2-lr-feat", type=float, default=1e-3, metavar="LR",
                        help="Feature-branch learning rate in Phase 2")
    parser.add_argument("--phase2-lr-gnn", type=float, default=5e-4, metavar="LR",
                        help="GNN learning rate in Phase 2 (kept low to preserve Phase 1 signal)")
    parser.add_argument("--weight-decay-p1", type=float, default=1e-2, metavar="WD",
                        help="Weight decay for Phase 1 AdamW (heavy to prevent MLP overfitting)")
    parser.add_argument("--weight-decay-p2", type=float, default=1e-4, metavar="WD",
                        help="Weight decay for Phase 2 AdamW")

    # ── LateFusionGNN training ────────────────────────────────────────────────
    parser.add_argument("--fusion-batch-size", type=int, default=32, metavar="N",
                        help="Batch size for LateFusionGNN training")
    parser.add_argument("--fusion-epochs", type=int, default=15, metavar="N",
                        help="Epochs to train the LateFusionGNN fusion head")
    parser.add_argument("--fusion-lr", type=float, default=1e-3, metavar="LR",
                        help="Learning rate for the fusion head optimizer")

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument("--hidden-dim", type=int, default=128, metavar="N",
                        help="Hidden dimension size for GATBlock layers")
    parser.add_argument("--heads", type=int, default=4, metavar="N",
                        help="Number of multi-head attention heads in GATv2Conv")
    parser.add_argument("--dropout", type=float, default=0.2, metavar="P",
                        help="Dropout probability inside GAT layers")

    # ── Dataset / splitting ───────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="Random seed for reproducible gene-level splits")
    parser.add_argument("--train-ratio", type=float, default=0.8, metavar="F",
                        help="Fraction of genes allocated to the training split")
    parser.add_argument("--val-ratio", type=float, default=0.1, metavar="F",
                        help="Fraction of genes allocated to the validation split")
    parser.add_argument("--window", type=int, default=300, metavar="N",
                        help="Half-width of the local sequence window around each SNP (±N nt)")

    args = parser.parse_args()

    # Derived checkpoint file paths — built from --checkpoint-dir
    args.expert_weights   = os.path.join(args.checkpoint_dir, "perturbation_gat_hpc2.pth")
    args.fusion_weights   = os.path.join(args.checkpoint_dir, "late_fusion_best.pth")
    args.best_gat_weights = os.path.join(args.checkpoint_dir, "perturbation_gat_best.pth")

    return args
