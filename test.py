import torch
import torch.nn as nn
import sys
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, confusion_matrix

from build_gnn_dataset import LncRNASiameseDataset
from GAT_model import PerturbationGAT
from train import gene_level_split


def evaluate_test_set():
    ROOT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
    MODEL_WEIGHTS = "perturbation_gat_best.pth"
    BS = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation on: {device}")

    dataset = LncRNASiameseDataset(root_dir=ROOT_DIR)

    # Use the EXACT same gene-level split (same seed=42)
    _, _, test_idx = gene_level_split(dataset)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    print(f"Test dataset size: {len(test_dataset)} samples.")
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    model = PerturbationGAT(node_in_dim=13, edge_dim=3, graph_feat_dim=16, hidden_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"Loaded '{MODEL_WEIGHTS}'")
    except FileNotFoundError:
        print(f"Error: '{MODEL_WEIGHTS}' not found. Run train.py first.")
        return

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            loss = criterion(logits, batch_data.y.view(-1, 1))
            test_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = batch_data.y.cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets)

    avg_test_loss = test_loss / len(test_loader)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    binary_preds = (all_preds >= 0.5).astype(int)

    accuracy = accuracy_score(all_targets, binary_preds)
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    cm = confusion_matrix(all_targets, binary_preds)

    try:
        auroc = roc_auc_score(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
    except ValueError:
        auroc, auprc = 0.0, 0.0

    print("\n" + "=" * 40)
    print("        FINAL TEST METRICS")
    print("=" * 40)
    print(f"Test Loss:        {avg_test_loss:.4f}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall (Sens):    {recall:.4f}")
    print(f"AUROC:            {auroc:.4f}")
    print(f"AUPRC:            {auprc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_test_set()
