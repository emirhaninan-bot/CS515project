import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool


class GATBlock(nn.Module):
    """Single GAT layer with residual connection and layer norm."""
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_dim, heads=heads, concat=True, edge_dim=edge_dim, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.residual = nn.Linear(in_dim, out_dim * heads) if in_dim != out_dim * heads else nn.Identity()
        
    def forward(self, x, edge_index, edge_attr):
        identity = self.residual(x)
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x + identity)
        x = F.leaky_relu(x)
        return x


class PerturbationGAT(nn.Module):
    """
    Hybrid Perturbation GAT: combines learned GNN embeddings with 
    handcrafted graph-level features proven effective by the RF baseline.
    
    The GNN learns spatial structural patterns, while the handcrafted features
    provide the classifier with direct access to aggregate perturbation statistics
    (total L1 delta, edge counts, etc.) that the GNN's message-passing struggles
    to extract on its own.
    """
    def __init__(self, node_in_dim=13, edge_dim=3, graph_feat_dim=16, 
                 hidden_dim=128, heads=4, dropout=0.2):
        super(PerturbationGAT, self).__init__()
        
        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16)
        )
        proj_edge_dim = 16
        
        # Input projection
        self.input_proj = nn.Linear(node_in_dim, hidden_dim * heads)
        
        # GAT blocks with residual connections
        self.block1 = GATBlock(hidden_dim * heads, hidden_dim, heads, proj_edge_dim, dropout)
        self.block2 = GATBlock(hidden_dim * heads, hidden_dim, heads, proj_edge_dim, dropout)
        
        # Final conv
        self.conv_final = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, 
                                     edge_dim=proj_edge_dim, dropout=dropout)
        self.norm_final = nn.LayerNorm(hidden_dim)
        
        # JK projection
        jk_dim = (hidden_dim * heads) * 2 + hidden_dim
        self.jk_proj = nn.Linear(jk_dim, hidden_dim)
        
        # Pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.pool_gate)
        
        # GNN output: attention_pool + max_pool = hidden_dim * 2
        self.gnn_out_dim = hidden_dim * 2
        gnn_out_dim = self.gnn_out_dim
        
        # Handcrafted feature processing
        self.feat_proj = nn.Sequential(
            nn.Linear(graph_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )
        
        # Classifier: GNN embedding + processed handcrafted features
        combined_dim = gnn_out_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data, use_gnn=True, return_embed=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        graph_features = data.graph_features.view(-1, self.feat_proj[0].in_features)
        
        if use_gnn:
            # Project edge features
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                edge_attr = self.edge_proj(edge_attr)
            
            # GNN forward
            x = F.leaky_relu(self.input_proj(x))
            x1 = self.block1(x, edge_index, edge_attr)
            x2 = self.block2(x1, edge_index, edge_attr)
            x3 = F.leaky_relu(self.norm_final(self.conv_final(x2, edge_index, edge_attr)))
            
            # JK aggregation
            x_jk = torch.cat([x1, x2, x3], dim=1)
            x_jk = self.jk_proj(x_jk)
            
            # Pooling
            x_attn = self.pool(x_jk, batch)
            x_max = global_max_pool(x_jk, batch)
            gnn_embed = torch.cat([x_attn, x_max], dim=1)
        else:
            # Skip GNN entirely — feed zeros so classifier learns from features only
            batch_size = graph_features.size(0)
            gnn_embed = torch.zeros(batch_size, self.gnn_out_dim, device=graph_features.device)
        
        # Process handcrafted features
        feat_embed = self.feat_proj(graph_features)
        
        # Combine GNN + handcrafted and classify
        combined = torch.cat([gnn_embed, feat_embed], dim=1)
        if return_embed:
            return combined
            
        logits = self.classifier(combined)
        
        return logits

class LateFusionGNN(nn.Module):
    def __init__(self, expert_weights_path, device):
        super().__init__()
        
        # 1. The Expert GNN (expects 14 handcrafted features)
        self.expert = PerturbationGAT(node_in_dim=13, edge_dim=3, graph_feat_dim=14, hidden_dim=128)
        
        # Load weights safely
        state_dict = torch.load(expert_weights_path, map_location=device)
        self.expert.load_state_dict(state_dict)
        
        # Freeze the expert completely
        for param in self.expert.parameters():
            param.requires_grad = False
            
        # 2. New Features Branch (Conservation + Delta G = 2 features)
        self.new_feat_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU()
        )
        
        # 3. Dynamic Attention Gate
        # Takes [expert_embed (288) + new_embed (16)] -> [weight_expert, weight_new]
        expert_embed_dim = self.expert.gnn_out_dim + 32 # 256 + 32 = 288
        
        self.gate_nn = nn.Sequential(
            nn.Linear(expert_embed_dim + 16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # 4. Final Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(expert_embed_dim + 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # The dataset provides 16 features total
        gf_all = data.graph_features.view(-1, 16)
        
        # Split features
        old_feats = gf_all[:, :14] # The 14 original structural features
        new_feats = gf_all[:, 14:] # Conservation (14) + Delta G (15)
        
        # Temporarily override graph_features for the expert
        data.graph_features = old_feats
        
        # The expert is frozen, no gradients needed for its internal ops
        with torch.no_grad():
            expert_embed = self.expert(data, return_embed=True)
            
        # Restore them just in case
        data.graph_features = gf_all.view(-1)
        
        # Process new features
        new_embed = self.new_feat_proj(new_feats)
        
        # Calculate Attention Gate
        concat_embed = torch.cat([expert_embed, new_embed], dim=1)
        alpha = self.gate_nn(concat_embed) # Shape: (Batch, 2)
        
        # Apply Attention Weights
        weighted_expert = expert_embed * alpha[:, 0].unsqueeze(1)
        weighted_new = new_embed * alpha[:, 1].unsqueeze(1)
        
        # Classify
        final_concat = torch.cat([weighted_expert, weighted_new], dim=1)
        logits = self.classifier(final_concat)
        
        return logits
