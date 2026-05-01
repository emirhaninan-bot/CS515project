import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from parameters import get_args
from data.build_gnn_dataset import LncRNASiameseDataset
from models.GAT_model import PerturbationGAT, LateFusionGNN


class FocalLoss(nn.Module):
    """Binary focal loss that down-weights easy negatives during training.

    Wraps BCEWithLogitsLoss with a modulating factor (1 - p_t)^gamma so that
    well-classified examples contribute less to the gradient, forcing the model
    to focus on hard, misclassified examples.

    Args:
        alpha (float): Overall loss scaling factor.
        gamma (float): Focusing exponent. Higher values suppress easy examples
            more aggressively (gamma=0 reduces to standard BCE).
        pos_weight (Tensor, optional): Per-class weight for the positive class
            passed to BCEWithLogitsLoss to handle label imbalance.
    """

    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        """Compute the mean focal loss over a batch.

        Args:
            inputs (Tensor): Raw logits of shape (N, 1).
            targets (Tensor): Binary ground-truth labels of shape (N, 1).

        Returns:
            Tensor: Scalar mean focal loss.
        """
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def gene_level_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split a dataset by gene so that all variants of one gene land in one partition.

    Grouping by gene prevents data leakage: if gene ACTA2 has 50 variants all 50
    go into the same partition, so the model cannot memorise gene-specific structural
    patterns instead of learning general perturbation rules.

    Args:
        dataset (LncRNASiameseDataset): Dataset whose ``pairs`` list contains
            gene-name metadata for every sample.
        train_ratio (float): Fraction of unique genes assigned to the training set.
        val_ratio (float): Fraction of unique genes assigned to the validation set.
            The remaining genes form the test set.
        seed (int): NumPy random seed for reproducible gene shuffling.

    Returns:
        tuple[list[int], list[int], list[int]]: (train_idx, val_idx, test_idx) —
            sample-level indices into ``dataset`` for each partition.
    """
    rng = np.random.RandomState(seed)

    gene_names = dataset.get_gene_names()
    gene_to_indices = defaultdict(list)
    for idx, gene in enumerate(gene_names):
        gene_to_indices[gene].append(idx)

    genes = list(gene_to_indices.keys())
    rng.shuffle(genes)

    n_genes = len(genes)
    n_train = int(n_genes * train_ratio)
    n_val   = int(n_genes * val_ratio)

    train_genes = set(genes[:n_train])
    val_genes   = set(genes[n_train:n_train + n_val])
    test_genes  = set(genes[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for gene in train_genes:
        train_idx.extend(gene_to_indices[gene])
    for gene in val_genes:
        val_idx.extend(gene_to_indices[gene])
    for gene in test_genes:
        test_idx.extend(gene_to_indices[gene])

    train_labels = [dataset.pairs[i]['label'] for i in train_idx]
    val_labels   = [dataset.pairs[i]['label'] for i in val_idx]
    test_labels  = [dataset.pairs[i]['label'] for i in test_idx]

    print("Gene-level split:")
    print(f"  Train: {len(train_genes)} genes, {len(train_idx)} samples "
          f"({int(sum(train_labels))} Pathogenic, {len(train_idx) - int(sum(train_labels))} Benign)")
    print(f"  Val:   {len(val_genes)} genes, {len(val_idx)} samples "
          f"({int(sum(val_labels))} Pathogenic, {len(val_idx) - int(sum(val_labels))} Benign)")
    print(f"  Test:  {len(test_genes)} genes, {len(test_idx)} samples "
          f"({int(sum(test_labels))} Pathogenic, {len(test_idx) - int(sum(test_labels))} Benign)")

    return train_idx, val_idx, test_idx


def run_training_loop(args):
    """Train the PerturbationGAT model using a two-phase curriculum.

    Phase 1 freezes the GNN and trains only the handcrafted-feature branch
    (feat_proj + classifier) to establish a strong feature baseline before the
    GNN is introduced. Phase 2 unfreezes the entire model but uses a lower
    learning rate for the GNN params so it can refine without erasing the signal
    learned in Phase 1.

    Args:
        args (argparse.Namespace): Parsed arguments from ``parameters.get_args()``.
            Relevant fields: data_dir, batch_size, phase1_epochs, phase2_epochs,
            phase1_lr, phase2_lr_feat, phase2_lr_gnn, weight_decay_p1,
            weight_decay_p2, hidden_dim, heads, dropout, seed, train_ratio,
            val_ratio, best_gat_weights.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    dataset = LncRNASiameseDataset(root_dir=args.data_dir)
    train_idx, val_idx, _ = gene_level_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=True)

    train_labels = [dataset.pairs[i]['label'] for i in train_idx]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    print(f"Class balance: {int(n_pos)} pathogenic / {int(n_neg)} benign "
          f"| pos_weight={pos_weight.item():.2f}")

    model = PerturbationGAT(
        node_in_dim=13, edge_dim=3, graph_feat_dim=16,
        hidden_dim=args.hidden_dim, heads=args.heads, dropout=args.dropout,
    ).to(device)
    criterion     = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    best_val_auroc = 0.0

    # ── Phase 1: Feature branch only (GNN frozen) ────────────────────────────
    print(f"\n=== PHASE 1: Feature branch only, GNN frozen ({args.phase1_epochs} epochs) ===")

    for name, param in model.named_parameters():
        param.requires_grad = ('feat_proj' in name or 'classifier' in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=args.weight_decay_p1,
    )

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            logits = model(batch_data, use_gnn=False)
            loss   = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"P1 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    # ── Phase 2: Full model, GNN unfrozen at lower LR ────────────────────────
    print(f"\n=== PHASE 2: Full model fine-tuning, GNN unfrozen ({args.phase2_epochs} epochs) ===")

    for param in model.parameters():
        param.requires_grad = True

    gnn_params, feat_params = [], []
    for name, param in model.named_parameters():
        if 'feat_proj' in name or 'classifier' in name:
            feat_params.append(param)
        else:
            gnn_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': gnn_params,  'lr': args.phase2_lr_gnn},
        {'params': feat_params, 'lr': args.phase2_lr_feat},
    ], weight_decay=args.weight_decay_p2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase2_epochs, eta_min=1e-6,
    )

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss   = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"P2 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                val_loss += criterion(logits, batch_data.y.view(-1, 1)).item()
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(batch_data.y.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        current_lr     = optimizer.param_groups[0]['lr']
        try:
            auroc = roc_auc_score(all_targets, all_preds)
            auprc = average_precision_score(all_targets, all_preds)
        except ValueError:
            auroc, auprc = 0.0, 0.0

        print(f"--- Epoch {epoch} (lr={current_lr:.6f}) ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val AUROC: {auroc:.4f} | Val AUPRC: {auprc:.4f}")

        if auroc > best_val_auroc:
            best_val_auroc = auroc
            torch.save(model.state_dict(), args.best_gat_weights)
            print(f">> Saved new best model! ({args.best_gat_weights})")

        print("---\n")


def train_late_fusion(args):
    """Train the LateFusionGNN fusion head on top of a frozen PerturbationGAT expert.

    The expert GNN is loaded from ``args.expert_weights`` and its parameters are
    frozen throughout training. Only the new-feature projection branch, the
    attention gate, and the final fusion classifier are updated. A 30% feature
    dropout on the conservation score (graph_features index 14) is applied per
    batch to prevent the model from over-relying on that single signal.

    Args:
        args (argparse.Namespace): Parsed arguments from ``parameters.get_args()``.
            Relevant fields: data_dir, expert_weights, fusion_batch_size,
            fusion_epochs, fusion_lr, weight_decay_p2, seed, train_ratio,
            val_ratio, fusion_weights.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Late Fusion Training on: {device}")

    dataset = LncRNASiameseDataset(root_dir=args.data_dir)
    train_idx, val_idx, _ = gene_level_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=args.fusion_batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=args.fusion_batch_size,
                              shuffle=False, drop_last=True)

    model = LateFusionGNN(expert_weights_path=args.expert_weights, device=device).to(device)

    y_train   = [dataset.pairs[i]['label'] for i in train_idx]
    n_pos     = sum(y_train)
    n_neg     = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    criterion       = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.fusion_lr, weight_decay=args.weight_decay_p2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.fusion_epochs, eta_min=1e-6,
    )

    best_val_auroc = 0.0
    print(f"\n=== Training Late Fusion Head ({args.fusion_epochs} epochs) ===")

    for epoch in range(1, args.fusion_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # 30% chance to zero-out the conservation score (feature index 14)
            if torch.rand(1).item() < 0.30:
                gf_view = batch_data.graph_features.view(-1, 16)
                gf_view[:, 14] = 0.0
                batch_data.graph_features = gf_view.view(-1)

            logits = model(batch_data)
            loss   = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_targets.extend(batch_data.y.cpu().numpy().flatten())

        val_auroc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch:2d}/{args.fusion_epochs} | "
              f"Loss: {total_loss / len(train_loader):.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), args.fusion_weights)
            print(f"   -> Saved new best fusion model! (AUROC: {val_auroc:.4f})")

    print(f"\nTraining complete. Best Late Fusion Val AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    run_training_loop(get_args())
