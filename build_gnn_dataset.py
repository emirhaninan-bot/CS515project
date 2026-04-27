import os
import glob
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import coalesce, add_self_loops
import pandas as pd

def load_fasta_to_dict(fasta_paths):
    """
    Loads one or more FASTA files into memory map: { 'HEADER_NAME': 'SEQUENCE' }
    """
    seq_dict = {}
    for path in fasta_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            header = None
            seq_parts = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if header is not None:
                        seq_dict[header] = "".join(seq_parts)
                    header = line[1:]  
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if header is not None:
                seq_dict[header] = "".join(seq_parts)
    return seq_dict


class LncRNASiameseDataset(Dataset):
    def __init__(self, root_dir, window=300, transform=None, pre_transform=None):
        """
        root_dir: directory containing the 4 matrix folders and fasta files.
        window: half-width of the local window around the mutation site (±window nt).
        """
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.window = window

        # Load all Sequences into Memory
        fasta_files = glob.glob(os.path.join(root_dir, "*.fasta"))
        print(f"Loading {len(fasta_files)} FASTA files into memory...")
        self.sequences = load_fasta_to_dict(fasta_files)

        # Discover pairs
        self.pairs = []
        self._discover_pairs('pathogenic', 1.0)
        self._discover_pairs('benign', 0.0)
        print(f"Validated {len(self.pairs)} complete WT/MT pairs.")
        
        # Extract gene names for gene-level splitting
        unique_genes = set(p['gene'] for p in self.pairs)
        print(f"Unique genes: {len(unique_genes)}")

        # Load Conservation Scores
        self.cons_dict = {}
        cons_path = os.path.join(root_dir, "conservation_features.csv")
        if os.path.exists(cons_path):
            cons_df = pd.read_csv(cons_path)
            self.cons_dict = dict(zip(cons_df['VariationID'], cons_df['phylop447_score']))
            print(f"Loaded {len(self.cons_dict)} conservation scores.")
        else:
            print("WARNING: conservation_features.csv not found. Scores will be 0.")
            
        # Load Delta G Features
        self.delta_g_dict = {}
        dg_path = os.path.join(root_dir, "delta_g_features.csv")
        if os.path.exists(dg_path):
            dg_df = pd.read_csv(dg_path)
            self.delta_g_dict = dict(zip(dg_df['mt_seq_id'], dg_df['delta_delta_g']))
            print(f"Loaded {len(self.delta_g_dict)} Delta G thermodynamic scores.")
        else:
            print("WARNING: delta_g_features.csv not found. Delta G will be 0.")

    def get_gene_names(self):
        """Returns list of gene names aligned with pair indices (for gene-level splitting)."""
        return [p['gene'] for p in self.pairs]

    def _discover_pairs(self, class_prefix, label):
        wt_dir = os.path.join(self.root_dir, f"{class_prefix}_wildtypes_matrices")
        mt_dir = os.path.join(self.root_dir, f"{class_prefix}_mutants_matrices")

        if not os.path.exists(wt_dir) or not os.path.exists(mt_dir):
            print(f"  WARNING: Missing directory for '{class_prefix}'. Skipping.")
            return

        mt_files = glob.glob(os.path.join(mt_dir, "*.npz"))
        matched = 0
        no_wt = 0
        no_fasta = 0

        for mt_file in mt_files:
            base_name = os.path.basename(mt_file)
            wt_base_name = base_name.replace("_MT.npz", "_WT.npz")
            wt_file = os.path.join(wt_dir, wt_base_name)

            if not os.path.exists(wt_file):
                no_wt += 1
                continue

            
            fasta_id_mt = base_name.replace(".npz", "").replace('\uf03a', ':')
            fasta_id_wt = wt_base_name.replace(".npz", "").replace('\uf03a', ':')

            # Extract gene name (first field before _VarID)
            gene_name = fasta_id_wt.split('_VarID')[0]

            if fasta_id_mt in self.sequences and fasta_id_wt in self.sequences:
                self.pairs.append({
                    'wt_matrix': wt_file,
                    'mt_matrix': mt_file,
                    'wt_seq_id': fasta_id_wt,
                    'mt_seq_id': fasta_id_mt,
                    'label': label,
                    'gene': gene_name
                })
                matched += 1
            else:
                no_fasta += 1

        print(f"  [{class_prefix}] NPZ files: {len(mt_files)} | Matched pairs: {matched} | Missing WT: {no_wt} | Missing FASTA: {no_fasta}")

    def len(self):
        return len(self.pairs)

    def _load_sparse_matrix(self, npz_path):
        """Loads a scipy CSR matrix from .npz file."""
        data = np.load(npz_path, allow_pickle=True)
        return sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

    def _extract_offset(self, seq_id):
        # Extract Offset value from ">NONHSAT000433.2_VarID:6772_Offset:504_MT"
        try:
            offset_str = seq_id.split("Offset:")[1].split("_")[0]
            return int(offset_str)
        except IndexError:
            return 0
            
    def _extract_varid(self, seq_id):
        try:
            var_str = seq_id.split("VarID:")[1].split("_")[0]
            try:
                return int(var_str)
            except ValueError:
                return var_str
        except IndexError:
            return -1

    def get(self, idx):
        pair = self.pairs[idx]
        offset = self._extract_offset(pair['wt_seq_id'])

        # ---- Load sequences ----
        wt_seq = self.sequences[pair['wt_seq_id']]
        mt_seq = self.sequences[pair['mt_seq_id']]
        seq_len = len(wt_seq)

        # ---- Local window around mutation ----
        start = max(0, offset - self.window)
        end = min(seq_len, offset + self.window + 1)
        local_offset = offset - start  # offset relative to cropped window

        wt_local = wt_seq[start:end]
        mt_local = mt_seq[start:end]

        # ---- Node features: [WT one-hot(4), MT one-hot(4), mut_flag(1), changed(1), dist_encoding(1)] = 11 dims ----
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
        n_nodes = end - start
        x = torch.zeros((n_nodes, 11), dtype=torch.float)

        for i in range(n_nodes):
            wt_base = wt_local[i].upper()
            mt_base = mt_local[i].upper()
            if wt_base in base_map:
                x[i, base_map[wt_base]] = 1.0       # WT one-hot (cols 0-3)
            if mt_base in base_map:
                x[i, 4 + base_map[mt_base]] = 1.0   # MT one-hot (cols 4-7)
            if i == local_offset:
                x[i, 8] = 1.0                        # mutation flag
            if wt_base != mt_base:
                x[i, 9] = 1.0                        # base changed flag
            # Positional encoding: distance decay from mutation site
            # Nodes closer to the SNP get higher values (1.0 at mutation, decays outward)
            x[i, 10] = 1.0 / (1.0 + abs(i - local_offset))

        # ---- Load structural matrices and crop to local window ----
        mat_wt = self._load_sparse_matrix(pair['wt_matrix'])
        mat_mt = self._load_sparse_matrix(pair['mt_matrix'])

        # Crop both matrices to the local window [start:end, start:end]
        mat_wt_local = mat_wt[start:end, start:end]
        mat_mt_local = mat_mt[start:end, start:end]

        # ---- Build edge set: BPPM + Covalent Backbone ----
        mat_union = (mat_wt_local + mat_mt_local)
        coo = mat_union.tocoo()

        bp_row = torch.tensor(coo.row, dtype=torch.long)
        bp_col = torch.tensor(coo.col, dtype=torch.long)

        # 1. Base Pair Edge Features
        wt_dense = mat_wt_local.toarray().astype(np.float32)
        mt_dense = mat_mt_local.toarray().astype(np.float32)
        
        wt_probs = wt_dense[coo.row, coo.col]
        mt_probs = mt_dense[coo.row, coo.col]
        delta = mt_probs - wt_probs
        
        threshold = 0.01
        mask = (wt_probs > threshold) & (mt_probs > threshold)
        bp_row = bp_row[mask]
        bp_col = bp_col[mask]
        wt_probs = wt_probs[mask]
        mt_probs = mt_probs[mask]
        delta = delta[mask]

        bp_edge_attr = np.stack([wt_probs, mt_probs, delta], axis=1)

        # 2. Add Covalent Backbone Edges (i -> i+1 and i+1 -> i)
        bb_row, bb_col = [], []
        for i in range(n_nodes - 1):
            bb_row.extend([i, i+1])
            bb_col.extend([i+1, i])
            
        if len(bb_row) > 0:
            bb_row_t = torch.tensor(bb_row, dtype=torch.long)
            bb_col_t = torch.tensor(bb_col, dtype=torch.long)
            
            # Backbone is perfectly stable: wt=1.0, mt=1.0, delta=0.0
            bb_edge_attr = np.tile([1.0, 1.0, 0.0], (len(bb_row), 1))
            
            # Combine Base Pairs and Backbone
            edge_index = torch.cat([torch.stack([bp_row, bp_col], dim=0), 
                                    torch.stack([bb_row_t, bb_col_t], dim=0)], dim=1)
            edge_attr = torch.from_numpy(np.vstack([bp_edge_attr, bb_edge_attr])).float()
        else:
            edge_index = torch.stack([bp_row, bp_col], dim=0)
            edge_attr = torch.from_numpy(bp_edge_attr).float()

        # ### FIX: REMOVE DUPLICATES
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')

        # ### FIX: ADD SELF LOOPS
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.0)

        # ---- Per-node edge delta aggregates (inject edge signal into node features) ----
        node_delta_sum = np.zeros(n_nodes, dtype=np.float32)
        node_delta_max = np.zeros(n_nodes, dtype=np.float32)
        abs_delta = np.abs(delta)
        
        # Use MASKED bp_row/bp_col (matching delta's length), not unmasked coo.row
        bp_row_np = bp_row.numpy()
        for e in range(len(bp_row_np)):
            r = bp_row_np[e]
            node_delta_sum[r] += abs_delta[e]
            node_delta_max[r] = max(node_delta_max[r], abs_delta[e])
        
        # Append as node features [cols 11, 12]
        x = torch.cat([
            x,
            torch.from_numpy(node_delta_sum).unsqueeze(1),
            torch.from_numpy(node_delta_max).unsqueeze(1)
        ], dim=1)  # Now 13 dims per node


        # ---- Graph-level handcrafted features ----
        # CRITICAL: Include GLOBAL features from full matrices (the strongest signal!)
        
        # Global features (full matrix - before cropping)
        global_diff = mat_mt - mat_wt
        global_abs_diff = abs(global_diff)
        global_l1 = global_abs_diff.sum()
        
        # Local features (cropped window)
        diff_local = mt_dense - wt_dense
        abs_diff_local = np.abs(diff_local)
        
        # Delta statistics from local window
        nonzero_deltas = diff_local[diff_local != 0]
        if len(nonzero_deltas) > 0:
            max_abs = np.max(np.abs(nonzero_deltas))
            mean_abs = np.mean(np.abs(nonzero_deltas))
            std_delta = np.std(nonzero_deltas)
        else:
            max_abs, mean_abs, std_delta = 0.0, 0.0, 0.0
        
        graph_features = np.array([
            # Global features (full matrix) — log-scaled for counts
            np.log1p(global_l1),                                       # 0: Global L1 diff
            np.log1p(global_abs_diff.nnz),                             # 1: Global changed edges
            np.log1p(float(mat_wt.nnz)),                               # 2: Global WT edges
            np.log1p(float(mat_mt.nnz)),                               # 3: Global MT edges
            np.log1p(float(abs(mat_wt.nnz - mat_mt.nnz))),             # 4: Global edge count diff
            # Local features (cropped window)
            np.log1p(abs_diff_local.sum()),                            # 5: Local L1 diff
            np.log1p(np.count_nonzero(abs_diff_local)),                # 6: Local changed edges
            np.log1p(float(mat_wt_local.nnz)),                        # 7: Local WT edges
            np.log1p(float(mat_mt_local.nnz)),                        # 8: Local MT edges
            # Delta statistics (raw — already small scale)
            float(max_abs),                                            # 9: Max absolute delta
            float(mean_abs),                                           # 10: Mean absolute delta
            float(std_delta),                                          # 11: Std of deltas
            np.log1p(float(np.sum(diff_local > 0))),                   # 12: Edges strengthened
            np.log1p(float(np.sum(diff_local < 0))),                   # 13: Edges weakened
            
            # Additional biological features
            float(self.cons_dict.get(self._extract_varid(pair['mt_seq_id']), 0.0)),  # 14: Conservation Score
            float(self.delta_g_dict.get(pair['mt_seq_id'], 0.0)),                    # 15: Delta Delta G
        ], dtype=np.float32)

        y = torch.tensor([pair['label']], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.graph_features = torch.from_numpy(graph_features)
        
        return data
