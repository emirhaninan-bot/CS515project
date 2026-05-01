import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from build_gnn_dataset import LncRNASiameseDataset
from GAT_model import PerturbationGAT, LateFusionGNN

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def gene_level_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset by GENE NAME, not by individual samples.
    
    This prevents data leakage: if gene ACTA2 has 50 variants, ALL 50 go
    into the same split. Otherwise the model memorizes gene-specific patterns
    rather than learning general structural perturbation rules.
    """
    rng = np.random.RandomState(seed)
    
    gene_names = dataset.get_gene_names()
    
    # Group sample indices by gene
    gene_to_indices = defaultdict(list)
    for idx, gene in enumerate(gene_names):
        gene_to_indices[gene].append(idx)
    
    # Shuffle genes (not samples)
    genes = list(gene_to_indices.keys())
    rng.shuffle(genes)
    
    # Split genes into train/val/test
    n_genes = len(genes)
    n_train = int(n_genes * train_ratio)
    n_val = int(n_genes * val_ratio)
    
    train_genes = set(genes[:n_train])
    val_genes = set(genes[n_train:n_train + n_val])
    test_genes = set(genes[n_train + n_val:])
    
    train_idx, val_idx, test_idx = [], [], []
    for gene in train_genes:
        train_idx.extend(gene_to_indices[gene])
    for gene in val_genes:
        val_idx.extend(gene_to_indices[gene])
    for gene in test_genes:
        test_idx.extend(gene_to_indices[gene])
    
    train_labels = [dataset.pairs[i]['label'] for i in train_idx]
    val_labels = [dataset.pairs[i]['label'] for i in val_idx]
    test_labels = [dataset.pairs[i]['label'] for i in test_idx]

    print(f"Gene-level split:")
    print(f"  Train: {len(train_genes)} genes, {len(train_idx)} samples ({int(sum(train_labels))} Pathogenic, {len(train_idx) - int(sum(train_labels))} Benign)")
    print(f"  Val:   {len(val_genes)} genes, {len(val_idx)} samples ({int(sum(val_labels))} Pathogenic, {len(val_idx) - int(sum(val_labels))} Benign)")
    print(f"  Test:  {len(test_genes)} genes, {len(test_idx)} samples ({int(sum(test_labels))} Pathogenic, {len(test_idx) - int(sum(test_labels))} Benign)")
    
    return train_idx, val_idx, test_idx


def run_training_loop():
    # 1. Configuration
    BS = 64
    EPOCHS = 30
    LR = 1e-3
    ROOT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 2. Build Dataset
    dataset = LncRNASiameseDataset(root_dir=ROOT_DIR)

    # Gene-level split (prevents data leakage)
    train_idx, val_idx, test_idx = gene_level_split(dataset)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=0, drop_last=True)

    # 3. Compute class weights for imbalanced data
    train_labels = [dataset.pairs[i]['label'] for i in train_idx]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    print(f"Class balance: {int(n_pos)} pathogenic / {int(n_neg)} benign | pos_weight={pos_weight.item():.2f}")

    # 4. Model
    model = PerturbationGAT(node_in_dim=13, edge_dim=3, graph_feat_dim=16, hidden_dim=128).to(device)
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    best_val_auroc = 0.0

    # =========================================================
    # TWO-PHASE TRAINING
    # Phase 1: Freeze GNN, train only feat_proj + classifier
    #          (establishes the proven handcrafted feature signal)
    # Phase 2: Unfreeze GNN with low LR so it can refine
    #          without destroying what Phase 1 learned
    # =========================================================
    
    PHASE1_EPOCHS = 5
    PHASE2_EPOCHS = 25
    
    # --- Phase 1: Feature-only warmup ---
    print("\n=== PHASE 1: Training feature branch only (GNN frozen) ===")
    
    # Freeze all GNN components
    for name, param in model.named_parameters():
        if 'feat_proj' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    
    # Use AdamW with heavy weight decay for Phase 1 to prevent MLP overfitting
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3, weight_decay=1e-2)
    
    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            logits = model(batch_data, use_gnn=False)
            loss = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"P1 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # --- Phase 2: Unfreeze GNN with low LR ---
    print("\n=== PHASE 2: Fine-tuning full model (GNN unfrozen) ===")
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    # Separate learning rates: GNN gets 10x lower LR than established feature branch
    gnn_params = []
    feat_params = []
    for name, param in model.named_parameters():
        if 'feat_proj' in name or 'classifier' in name:
            feat_params.append(param)
        else:
            gnn_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': gnn_params, 'lr': 5e-4},       # GNN: low LR
        {'params': feat_params, 'lr': 1e-3},       # Features: moderate LR
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    # 5. Training Loop (Phase 2)
    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            logits = model(batch_data)
            loss = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"P2 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        # 6. Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                loss = criterion(logits, batch_data.y.view(-1, 1))
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = batch_data.y.cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
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
            torch.save(model.state_dict(), 'perturbation_gat_best.pth')
            print(">> Saved new best model! (perturbation_gat_best.pth)")

        print("---\n")


def train_late_fusion():
    ROOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "."
    EXPERT_WEIGHTS = "perturbation_gat_hpc2.pth"
    BS = 32
    EPOCHS = 15 # Because the GNN is frozen, we only need a few epochs!

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Late Fusion Training on: {device}")

    # 1. Load Dataset
    dataset = LncRNASiameseDataset(root_dir=ROOT_DIR)
    train_idx, val_idx, _ = gene_level_split(dataset)

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False, drop_last=True)

    # 2. Setup Model
    model = LateFusionGNN(expert_weights_path=EXPERT_WEIGHTS, device=device).to(device)

    # Calculate class weights
    y_train = [dataset.pairs[i]['label'] for i in train_idx]
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    
    # We only train the new layers!
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_auroc = 0.0

    print(f"\n=== Training Late Fusion Head ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # Feature Dropout: 30% chance to blackout conservation score
            if torch.rand(1).item() < 0.30:
                gf_view = batch_data.graph_features.view(-1, 16)
                gf_view[:, 14] = 0.0
                batch_data.graph_features = gf_view.view(-1)

            logits = model(batch_data)
            loss = criterion(logits, batch_data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_targets.extend(batch_data.y.cpu().numpy().flatten())

        val_auroc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), "late_fusion_best.pth")
            print(f"   -> Saved new best fusion model! (AUROC: {val_auroc:.4f})")

    print(f"\nTraining complete. Best Late Fusion Val AUROC: {best_val_auroc:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fusion":
        train_late_fusion()
    else:
        run_training_loop()
