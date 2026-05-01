import os
import glob
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import coalesce, add_self_loops
import pandas as pd


def load_fasta_to_dict(fasta_paths):
    """Load one or more FASTA files into a header-to-sequence dictionary.

    Multi-line sequences are concatenated. The leading ``>`` of each header line
    is stripped so the key is the bare sequence identifier.

    Args:
        fasta_paths (list[str]): Paths to FASTA files to load. Non-existent
            paths are silently skipped.

    Returns:
        dict[str, str]: Mapping from sequence header to nucleotide sequence string.
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
                    header    = line[1:]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if header is not None:
                seq_dict[header] = "".join(seq_parts)
    return seq_dict


class LncRNASiameseDataset(Dataset):
    """PyG Dataset of wild-type / mutant lncRNA graph pairs for pathogenicity classification.

    Each sample is a siamese pair: wild-type and mutant RNA are represented as a
    single graph whose nodes encode nucleotide identity for both sequences and whose
    edges come from the union of their base-pair probability matrices. Graph-level
    handcrafted features summarise global and local structural perturbation statistics
    alongside phylogenetic conservation and thermodynamic stability changes.

    Directory layout expected under ``root_dir``::

        root_dir/
        ├── *_wildtypes_matrices/   # scipy CSR matrices (.npz) for WT sequences
        ├── *_mutants_matrices/     # scipy CSR matrices (.npz) for MT sequences
        ├── *.fasta                 # WT and MT sequences
        ├── conservation_features.csv
        └── delta_g_features.csv

    Args:
        root_dir (str): Path to the directory described above.
        window (int): Half-width of the local sequence window around each mutation
            site (±window nucleotides). Nodes outside this window are discarded to
            keep graphs tractable.
        transform: Optional per-sample transform applied after ``get()``.
        pre_transform: Optional transform applied once during dataset construction.
    """

    def __init__(self, root_dir, window=300, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.window   = window

        fasta_files = glob.glob(os.path.join(root_dir, "*.fasta"))
        print(f"Loading {len(fasta_files)} FASTA files into memory...")
        self.sequences = load_fasta_to_dict(fasta_files)

        self.pairs = []
        self._discover_pairs('pathogenic', 1.0)
        self._discover_pairs('benign', 0.0)
        print(f"Validated {len(self.pairs)} complete WT/MT pairs.")

        unique_genes = set(p['gene'] for p in self.pairs)
        print(f"Unique genes: {len(unique_genes)}")

        self.cons_dict = {}
        cons_path = os.path.join(root_dir, "conservation_features.csv")
        if os.path.exists(cons_path):
            cons_df = pd.read_csv(cons_path)
            self.cons_dict = dict(zip(cons_df['VariationID'], cons_df['phylop447_score']))
            print(f"Loaded {len(self.cons_dict)} conservation scores.")
        else:
            print("WARNING: conservation_features.csv not found. Scores will be 0.")

        self.delta_g_dict = {}
        dg_path = os.path.join(root_dir, "delta_g_features.csv")
        if os.path.exists(dg_path):
            dg_df = pd.read_csv(dg_path)
            self.delta_g_dict = dict(zip(dg_df['mt_seq_id'], dg_df['delta_delta_g']))
            print(f"Loaded {len(self.delta_g_dict)} Delta G thermodynamic scores.")
        else:
            print("WARNING: delta_g_features.csv not found. Delta G will be 0.")

    def get_gene_names(self):
        """Return the gene name for every sample, aligned with pair indices.

        Used by ``train.gene_level_split`` to group samples by gene before
        partitioning, preventing variants of the same gene from appearing in
        multiple splits.

        Returns:
            list[str]: Gene name for each index in ``self.pairs``.
        """
        return [p['gene'] for p in self.pairs]

    def _discover_pairs(self, class_prefix, label):
        """Scan matrix directories to find valid WT/MT pairs for one class.

        A pair is valid when both the WT and MT .npz files exist on disk and
        both corresponding FASTA sequences are present in ``self.sequences``.
        Valid pairs are appended to ``self.pairs``.

        Args:
            class_prefix (str): Either ``'pathogenic'`` or ``'benign'``.
            label (float): Class label to store with each pair (1.0 or 0.0).
        """
        wt_dir = os.path.join(self.root_dir, f"{class_prefix}_wildtypes_matrices")
        mt_dir = os.path.join(self.root_dir, f"{class_prefix}_mutants_matrices")

        if not os.path.exists(wt_dir) or not os.path.exists(mt_dir):
            print(f"  WARNING: Missing directory for '{class_prefix}'. Skipping.")
            return

        mt_files = glob.glob(os.path.join(mt_dir, "*.npz"))
        matched = no_wt = no_fasta = 0

        for mt_file in mt_files:
            base_name    = os.path.basename(mt_file)
            wt_base_name = base_name.replace("_MT.npz", "_WT.npz")
            wt_file      = os.path.join(wt_dir, wt_base_name)

            if not os.path.exists(wt_file):
                no_wt += 1
                continue

            fasta_id_mt = base_name.replace(".npz", "").replace('', ':')
            fasta_id_wt = wt_base_name.replace(".npz", "").replace('', ':')

            gene_name = fasta_id_wt.split('_VarID')[0]

            if fasta_id_mt in self.sequences and fasta_id_wt in self.sequences:
                self.pairs.append({
                    'wt_matrix': wt_file,
                    'mt_matrix': mt_file,
                    'wt_seq_id': fasta_id_wt,
                    'mt_seq_id': fasta_id_mt,
                    'label':     label,
                    'gene':      gene_name,
                })
                matched += 1
            else:
                no_fasta += 1

        print(f"  [{class_prefix}] NPZ files: {len(mt_files)} | "
              f"Matched pairs: {matched} | Missing WT: {no_wt} | Missing FASTA: {no_fasta}")

    def len(self):
        """Return the total number of WT/MT pairs in the dataset."""
        return len(self.pairs)

    def _load_sparse_matrix(self, npz_path):
        """Load a scipy CSR matrix from a compressed .npz file.

        Args:
            npz_path (str): Path to the .npz file storing ``data``, ``indices``,
                ``indptr``, and ``shape`` arrays produced by
                ``scipy.sparse.save_npz``.

        Returns:
            scipy.sparse.csr_matrix: The reconstructed sparse matrix.
        """
        data = np.load(npz_path, allow_pickle=True)
        return sp.csr_matrix(
            (data['data'], data['indices'], data['indptr']), shape=data['shape']
        )

    def _extract_offset(self, seq_id):
        """Extract the mutation-site offset from a sequence identifier.

        Sequence IDs follow the pattern
        ``GENE_VarID:N_Offset:M_MT`` where M is the 0-based nucleotide position
        of the mutation within the full sequence.

        Args:
            seq_id (str): Sequence header string.

        Returns:
            int: Mutation offset, or 0 if the tag is absent.
        """
        try:
            return int(seq_id.split("Offset:")[1].split("_")[0])
        except IndexError:
            return 0

    def _extract_varid(self, seq_id):
        """Extract the numeric or string variant ID from a sequence identifier.

        Args:
            seq_id (str): Sequence header string containing ``VarID:N``.

        Returns:
            int | str: Parsed variant ID, or -1 if the tag is absent.
        """
        try:
            var_str = seq_id.split("VarID:")[1].split("_")[0]
            try:
                return int(var_str)
            except ValueError:
                return var_str
        except IndexError:
            return -1

    def get(self, idx):
        """Build and return the PyG Data object for a single WT/MT pair.

        Processing steps:
        1. Load WT and MT sequences and crop to a ±window region around the SNP.
        2. Build node features: WT one-hot (4), MT one-hot (4), mutation flag (1),
           base-change flag (1), distance decay (1), per-node delta-sum (1),
           per-node delta-max (1) — total 13 dimensions.
        3. Build edge set from the union of WT and MT base-pair probability matrices,
           filtered to pairs where both probabilities exceed 0.01, plus covalent
           backbone edges (i→i+1).
        4. Compute 16 graph-level handcrafted features (global/local structural
           statistics, conservation score, delta-delta-G).

        Args:
            idx (int): Index into ``self.pairs``.

        Returns:
            torch_geometric.data.Data: Graph object with attributes ``x``
                (N × 13 node features), ``edge_index`` (2 × E), ``edge_attr``
                (E × 3), ``y`` (scalar label), and ``graph_features`` (16,).
        """
        pair   = self.pairs[idx]
        offset = self._extract_offset(pair['wt_seq_id'])

        wt_seq  = self.sequences[pair['wt_seq_id']]
        mt_seq  = self.sequences[pair['mt_seq_id']]
        seq_len = len(wt_seq)

        start        = max(0, offset - self.window)
        end          = min(seq_len, offset + self.window + 1)
        local_offset = offset - start

        wt_local = wt_seq[start:end]
        mt_local = mt_seq[start:end]

        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
        n_nodes  = end - start
        x        = torch.zeros((n_nodes, 11), dtype=torch.float)

        for i in range(n_nodes):
            wt_base = wt_local[i].upper()
            mt_base = mt_local[i].upper()
            if wt_base in base_map:
                x[i, base_map[wt_base]]     = 1.0
            if mt_base in base_map:
                x[i, 4 + base_map[mt_base]] = 1.0
            if i == local_offset:
                x[i, 8] = 1.0
            if wt_base != mt_base:
                x[i, 9] = 1.0
            x[i, 10] = 1.0 / (1.0 + abs(i - local_offset))

        mat_wt       = self._load_sparse_matrix(pair['wt_matrix'])
        mat_mt       = self._load_sparse_matrix(pair['mt_matrix'])
        mat_wt_local = mat_wt[start:end, start:end]
        mat_mt_local = mat_mt[start:end, start:end]

        mat_union = mat_wt_local + mat_mt_local
        coo       = mat_union.tocoo()

        bp_row = torch.tensor(coo.row, dtype=torch.long)
        bp_col = torch.tensor(coo.col, dtype=torch.long)

        wt_dense = mat_wt_local.toarray().astype(np.float32)
        mt_dense = mat_mt_local.toarray().astype(np.float32)

        wt_probs = wt_dense[coo.row, coo.col]
        mt_probs = mt_dense[coo.row, coo.col]
        delta    = mt_probs - wt_probs

        threshold = 0.01
        mask     = (wt_probs > threshold) & (mt_probs > threshold)
        bp_row   = bp_row[mask]
        bp_col   = bp_col[mask]
        wt_probs = wt_probs[mask]
        mt_probs = mt_probs[mask]
        delta    = delta[mask]

        bp_edge_attr = np.stack([wt_probs, mt_probs, delta], axis=1)

        bb_row, bb_col = [], []
        for i in range(n_nodes - 1):
            bb_row.extend([i, i + 1])
            bb_col.extend([i + 1, i])

        if len(bb_row) > 0:
            bb_row_t     = torch.tensor(bb_row, dtype=torch.long)
            bb_col_t     = torch.tensor(bb_col, dtype=torch.long)
            bb_edge_attr = np.tile([1.0, 1.0, 0.0], (len(bb_row), 1))

            edge_index = torch.cat(
                [torch.stack([bp_row, bp_col], dim=0),
                 torch.stack([bb_row_t, bb_col_t], dim=0)], dim=1
            )
            edge_attr = torch.from_numpy(
                np.vstack([bp_edge_attr, bb_edge_attr])
            ).float()
        else:
            edge_index = torch.stack([bp_row, bp_col], dim=0)
            edge_attr  = torch.from_numpy(bp_edge_attr).float()

        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr,
                                               fill_value=0.0)

        node_delta_sum = np.zeros(n_nodes, dtype=np.float32)
        node_delta_max = np.zeros(n_nodes, dtype=np.float32)
        abs_delta      = np.abs(delta)
        bp_row_np      = bp_row.numpy()
        for e in range(len(bp_row_np)):
            r = bp_row_np[e]
            node_delta_sum[r] += abs_delta[e]
            node_delta_max[r]  = max(node_delta_max[r], abs_delta[e])

        x = torch.cat([
            x,
            torch.from_numpy(node_delta_sum).unsqueeze(1),
            torch.from_numpy(node_delta_max).unsqueeze(1),
        ], dim=1)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

        global_diff     = mat_mt - mat_wt
        global_abs_diff = abs(global_diff)
        global_l1       = global_abs_diff.sum()

        diff_local     = mt_dense - wt_dense
        abs_diff_local = np.abs(diff_local)

        nonzero_deltas = diff_local[diff_local != 0]
        if len(nonzero_deltas) > 0:
            max_abs  = np.max(np.abs(nonzero_deltas))
            mean_abs = np.mean(np.abs(nonzero_deltas))
            std_delta = np.std(nonzero_deltas)
        else:
            max_abs = mean_abs = std_delta = 0.0

        graph_features = np.array([
            np.log1p(global_l1),                                        # 0: Global L1 diff
            np.log1p(global_abs_diff.nnz),                              # 1: Global changed edges
            np.log1p(float(mat_wt.nnz)),                                # 2: Global WT edges
            np.log1p(float(mat_mt.nnz)),                                # 3: Global MT edges
            np.log1p(float(abs(mat_wt.nnz - mat_mt.nnz))),              # 4: Global edge count diff
            np.log1p(abs_diff_local.sum()),                             # 5: Local L1 diff
            np.log1p(np.count_nonzero(abs_diff_local)),                 # 6: Local changed edges
            np.log1p(float(mat_wt_local.nnz)),                         # 7: Local WT edges
            np.log1p(float(mat_mt_local.nnz)),                         # 8: Local MT edges
            float(max_abs),                                             # 9: Max absolute delta
            float(mean_abs),                                            # 10: Mean absolute delta
            float(std_delta),                                           # 11: Std of deltas
            np.log1p(float(np.sum(diff_local > 0))),                    # 12: Edges strengthened
            np.log1p(float(np.sum(diff_local < 0))),                    # 13: Edges weakened
            float(self.cons_dict.get(
                self._extract_varid(pair['mt_seq_id']), 0.0)),          # 14: Conservation score
            float(self.delta_g_dict.get(pair['mt_seq_id'], 0.0)),       # 15: Delta-delta-G
        ], dtype=np.float32)

        y    = torch.tensor([pair['label']], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.graph_features = torch.from_numpy(graph_features)

        return data
