import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool


class GATBlock(nn.Module):
    """Single GATv2 attention layer with residual connection and layer normalisation.

    Applies one GATv2Conv step, adds the input via a residual projection (linear if
    dimensions differ, identity otherwise), normalises with LayerNorm, and activates
    with LeakyReLU.

    Args:
        in_dim (int): Input node-feature dimension.
        out_dim (int): Per-head output dimension. Actual output width is
            ``out_dim * heads`` because ``concat=True``.
        heads (int): Number of attention heads.
        edge_dim (int): Edge-feature dimension expected by GATv2Conv.
        dropout (float): Attention-coefficient dropout probability.
    """

    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout):
        super().__init__()
        self.conv     = GATv2Conv(in_dim, out_dim, heads=heads, concat=True,
                                  edge_dim=edge_dim, dropout=dropout)
        self.norm     = nn.LayerNorm(out_dim * heads)
        self.residual = (nn.Linear(in_dim, out_dim * heads)
                         if in_dim != out_dim * heads else nn.Identity())

    def forward(self, x, edge_index, edge_attr):
        """Apply one GATv2 step with residual addition and layer norm.

        Args:
            x (Tensor): Node features of shape (N, in_dim).
            edge_index (LongTensor): Graph connectivity of shape (2, E).
            edge_attr (Tensor): Edge features of shape (E, edge_dim).

        Returns:
            Tensor: Updated node features of shape (N, out_dim * heads).
        """
        identity = self.residual(x)
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x + identity)
        x = F.leaky_relu(x)
        return x


class PerturbationGAT(nn.Module):
    """Hybrid GNN that combines learned GATv2 embeddings with handcrafted graph features.

    Architecture overview:
    - Edge-feature projection (edge_dim → 16).
    - Input node projection (node_in_dim → hidden_dim * heads).
    - Two GATBlock layers with residual connections.
    - A final single-head GATv2Conv layer.
    - Jumping-knowledge (JK) concatenation of all three intermediate representations,
      projected down to hidden_dim.
    - Dual pooling: GlobalAttention + global_max_pool, concatenated → gnn_out_dim.
    - Handcrafted-feature MLP (graph_feat_dim → 32).
    - Classifier MLP over [gnn_embed ‖ feat_embed] → logit.

    The GNN learns spatial structural patterns from the RNA base-pair probability
    matrix, while the handcrafted features give the classifier direct access to
    aggregate perturbation statistics (total L1 delta, edge counts, etc.) that
    message-passing alone struggles to capture.

    Args:
        node_in_dim (int): Number of input node features.
        edge_dim (int): Number of edge features.
        graph_feat_dim (int): Number of graph-level handcrafted features.
        hidden_dim (int): Hidden dimension per attention head.
        heads (int): Number of attention heads for GATBlock layers.
        dropout (float): Dropout probability in GATv2Conv attention weights.
    """

    def __init__(self, node_in_dim=13, edge_dim=3, graph_feat_dim=16,
                 hidden_dim=128, heads=4, dropout=0.2):
        super(PerturbationGAT, self).__init__()

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
        )
        proj_edge_dim = 16

        self.input_proj = nn.Linear(node_in_dim, hidden_dim * heads)

        self.block1 = GATBlock(hidden_dim * heads, hidden_dim, heads, proj_edge_dim, dropout)
        self.block2 = GATBlock(hidden_dim * heads, hidden_dim, heads, proj_edge_dim, dropout)

        self.conv_final = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False,
                                    edge_dim=proj_edge_dim, dropout=dropout)
        self.norm_final = nn.LayerNorm(hidden_dim)

        jk_dim = (hidden_dim * heads) * 2 + hidden_dim
        self.jk_proj = nn.Linear(jk_dim, hidden_dim)

        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool = GlobalAttention(gate_nn=self.pool_gate)

        self.gnn_out_dim = hidden_dim * 2  # attention_pool + max_pool concatenated

        self.feat_proj = nn.Sequential(
            nn.Linear(graph_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )

        combined_dim = self.gnn_out_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, use_gnn=True, return_embed=False):
        """Forward pass over a PyG batch.

        Args:
            data (torch_geometric.data.Batch): Batched graph data with attributes
                ``x``, ``edge_index``, ``edge_attr``, ``batch``, and
                ``graph_features`` (stored flat as batch_size * graph_feat_dim).
            use_gnn (bool): When False the GNN sub-network is bypassed and replaced
                with zeros, so only the handcrafted features are active. Used in
                Phase 1 training to warm up the feature branch in isolation.
            return_embed (bool): When True return the combined embedding vector
                [gnn_embed ‖ feat_embed] instead of the final logit. Used by
                LateFusionGNN to extract the frozen expert representation.

        Returns:
            Tensor: Logits of shape (batch_size, 1), or the combined embedding of
                shape (batch_size, gnn_out_dim + 32) when ``return_embed=True``.
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        graph_features = data.graph_features.view(-1, self.feat_proj[0].in_features)

        if use_gnn:
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                edge_attr = self.edge_proj(edge_attr)

            x  = F.leaky_relu(self.input_proj(x))
            x1 = self.block1(x, edge_index, edge_attr)
            x2 = self.block2(x1, edge_index, edge_attr)
            x3 = F.leaky_relu(self.norm_final(self.conv_final(x2, edge_index, edge_attr)))

            x_jk      = self.jk_proj(torch.cat([x1, x2, x3], dim=1))
            x_attn    = self.pool(x_jk, batch)
            x_max     = global_max_pool(x_jk, batch)
            gnn_embed = torch.cat([x_attn, x_max], dim=1)
        else:
            gnn_embed = torch.zeros(graph_features.size(0), self.gnn_out_dim,
                                    device=graph_features.device)

        feat_embed = self.feat_proj(graph_features)
        combined   = torch.cat([gnn_embed, feat_embed], dim=1)

        if return_embed:
            return combined

        return self.classifier(combined)


class LateFusionGNN(nn.Module):
    """Late-fusion model that extends a frozen PerturbationGAT with new biological features.

    Architecture:
    - Expert branch: frozen PerturbationGAT loaded from a checkpoint. Its combined
      embedding [gnn_embed ‖ feat_embed] has dimension ``expert_embed_dim`` (288).
    - New-feature branch: small MLP over 2 extra features — phylogenetic conservation
      score and delta-delta-G thermodynamic stability — projected to 16 dimensions.
    - Attention gate: takes [expert_embed ‖ new_embed] and produces two softmax
      weights to dynamically re-weight each branch per sample.
    - Fusion classifier: operates on the weighted concatenation → logit.

    Args:
        expert_weights_path (str): Path to the saved PerturbationGAT state-dict (.pth).
        device (torch.device): Device on which to load the expert weights.
    """

    def __init__(self, expert_weights_path, device):
        super().__init__()

        self.expert = PerturbationGAT(node_in_dim=13, edge_dim=3,
                                      graph_feat_dim=14, hidden_dim=128)
        self.expert.load_state_dict(torch.load(expert_weights_path, map_location=device))
        for param in self.expert.parameters():
            param.requires_grad = False

        self.new_feat_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
        )

        expert_embed_dim = self.expert.gnn_out_dim + 32  # 256 + 32 = 288

        self.gate_nn = nn.Sequential(
            nn.Linear(expert_embed_dim + 16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(expert_embed_dim + 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        """Fuse expert and new-feature embeddings via a learned attention gate.

        Splits ``data.graph_features`` (16 features per sample) into the 14
        original structural features for the expert and the 2 new biological
        features. Applies the attention gate to dynamically weight each branch
        before the final classification.

        Args:
            data (torch_geometric.data.Batch): Batched graph data. Requires
                ``graph_features`` stored flat as (batch_size * 16,).

        Returns:
            Tensor: Logits of shape (batch_size, 1).
        """
        gf_all    = data.graph_features.view(-1, 16)
        old_feats = gf_all[:, :14]   # original 14 structural features → expert
        new_feats = gf_all[:, 14:]   # conservation score + delta-delta-G

        data.graph_features = old_feats
        with torch.no_grad():
            expert_embed = self.expert(data, return_embed=True)
        data.graph_features = gf_all.view(-1)

        new_embed    = self.new_feat_proj(new_feats)
        concat_embed = torch.cat([expert_embed, new_embed], dim=1)

        alpha           = self.gate_nn(concat_embed)           # (B, 2)
        weighted_expert = expert_embed * alpha[:, 0].unsqueeze(1)
        weighted_new    = new_embed    * alpha[:, 1].unsqueeze(1)

        return self.classifier(torch.cat([weighted_expert, weighted_new], dim=1))
