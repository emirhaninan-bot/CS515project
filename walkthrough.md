# How the GNN Learns from RNA Structure

## Step 1: The Raw Data — Base Pair Probability Matrices (BPPM)

LinearPartition predicts the 2D structure of an RNA sequence as a **matrix** where entry `(i, j)` = probability that nucleotide `i` pairs with nucleotide `j`.

For every variant, we have TWO matrices:
- **WT Matrix**: Structure of the original (healthy) RNA
- **MT Matrix**: Structure after the SNP mutation

Example for a 5-nucleotide RNA:
```
WT Matrix:              MT Matrix:
    A   C   G   U   A      A   C   G   U   A
A [ .   .   .  0.9  . ]  A [ .   .   .  0.2  . ]   ← Pair A-U was 0.9, now 0.2!
C [ .   .  0.8  .   . ]  C [ .   .  0.7  .   . ]   ← Pair C-G barely changed
G [ .  0.8  .   .   . ]  G [ .  0.7  .   .   . ]
U [0.9  .   .   .   . ]  U [0.2  .   .   .   . ]
A [ .   .   .   .   . ]  A [ .   .   .   .   . ]
```

The mutation **destroyed** the A-U base pair (0.9 → 0.2). This is a structural perturbation!

---

## Step 2: Building the Graph

We convert these matrices into a **graph** where:
- **Each node** = one nucleotide in the local window (±300 nt around the mutation)
- **Each edge** = a base pair that exists in WT, MT, or both

### Node Features (13 dimensions per nucleotide)
```
Node i = [A, C, G, U,  A, C, G, U,  mut_flag, changed, dist_decay, Δ_sum, Δ_max]
          ├─WT one-hot─┤├─MT one-hot─┤                               ├─edge stats─┤
```
- **WT/MT one-hot (8):** What base is at this position before and after mutation
- **mut_flag (1):** Is this THE mutation site? (1.0 at the SNP, 0.0 everywhere else)
- **changed (1):** Did the base change at this position?
- **dist_decay (1):** `1/(1 + distance from SNP)` — nodes near the mutation get higher values
- **Δ_sum, Δ_max (2):** Aggregated edge perturbation at this node (how much did this nucleotide's base pairs change?)

### Edge Features (3 dimensions per base pair)
```
Edge (i,j) = [wt_prob, mt_prob, delta]
              0.90     0.20    -0.70    ← "This pair was destroyed"
              0.80     0.82    +0.02    ← "This pair is stable"
```

### Backbone Edges
We also add covalent backbone edges (i → i+1) representing the physical RNA chain. These have features `[1.0, 1.0, 0.0]` (always stable, never change).

---

## Step 3: Message Passing (GATv2Conv)

This is where the actual learning happens. The GNN runs **3 rounds** of message passing.

### What is Message Passing?
Every node "talks" to its neighbors (the nucleotides it base-pairs with). In each round:

1. **Each node looks at its neighbors' features AND the edge features connecting them**
2. **It calculates an "attention score"** — how important is each neighbor?
3. **It aggregates the weighted messages** and updates its own representation

### Concretely (GATv2Conv with 4 attention heads):

```
For node i, looking at neighbor j connected by edge (i,j):

Step 1: Project edge features
   edge_embed = MLP([wt_prob=0.9, mt_prob=0.2, delta=-0.7])  →  16-dim vector

Step 2: Calculate attention
   attention_ij = LeakyReLU(W * [node_i || node_j || edge_embed])
   "How important is this broken base pair to node i?"

Step 3: Normalize attention across all neighbors
   α_ij = softmax(attention_ij)  over all neighbors of i

Step 4: Update node i
   node_i_new = Σ (α_ij * W_value * node_j)
```

### Why This Learns Structure:

Imagine node `A` at position 0, which pairs with node `U` at position 3.

**Round 1:** Node A receives a message from node U saying *"Our base pair was destroyed (delta = -0.7)."* Node A's representation now encodes: *"I lost my main structural partner."*

**Round 2:** Node A's neighbors receive A's updated representation. They now know: *"My neighbor A lost its base pair."* Information propagates outward.

**Round 3:** Nodes 2 hops away now know about the structural damage. The network builds a **global understanding** of how the perturbation ripples through the structure.

### Residual Connections & Layer Norm
After each round, we add the input back (residual connection) and normalize:
```python
x = LayerNorm(x_new + x_old)  # Prevents gradient vanishing
```

### Jumping Knowledge (JK) Aggregation
We concatenate outputs from ALL 3 rounds:
```python
x_jk = [round1_output || round2_output || round3_output]
x_jk = Linear(x_jk)  # Project back to 128 dims
```
This lets the classifier see BOTH local damage (round 1) and global ripple effects (round 3).

---

## Step 4: Pooling — From Nodes to Graph

After message passing, we have ~600 node vectors (one per nucleotide). We need ONE vector to represent the entire RNA.

### Attention Pooling
A small MLP learns which nodes are most important:
```python
importance_i = MLP(node_i)  → scalar
weights = softmax(importance_scores)  → which nodes matter?
graph_embed_attn = Σ (weight_i * node_i)
```
The model learns to **pay attention to the mutation site** and heavily perturbed regions.

### Max Pooling
```python
graph_embed_max = max(node_1, node_2, ..., node_600)  # element-wise max
```
Captures the **most extreme** perturbation signal anywhere in the RNA.

### Combined: 256-dim GNN embedding
```python
gnn_embed = [attention_pool (128) || max_pool (128)]  = 256 dims
```

---

## Step 5: Handcrafted Features Branch

In parallel, the 14 graph-level statistics bypass the GNN entirely:
```python
feat_embed = MLP([global_L1, changed_edges, ..., edges_weakened])  → 32 dims
```
These provide the classifier with aggregate perturbation statistics that are hard for message-passing to extract (e.g., total number of changed edges globally).

---

## Step 6: Classification

```python
combined = [gnn_embed (256) || feat_embed (32)]  = 288 dims

logit = Classifier_MLP(combined)  → 1 scalar
prediction = sigmoid(logit)  → 0.0 (Benign) to 1.0 (Pathogenic)
```

---

## What the Model Actually Learns

After training, the GNN has learned patterns like:

1. **"Cascade Collapse"**: If the mutation destroys one key base pair AND that destruction propagates to nearby pairs (visible in rounds 2-3), the RNA structure is fundamentally broken → **Pathogenic**

2. **"Isolated Wobble"**: If the mutation changes one base pair but surrounding pairs remain stable (delta ≈ 0 for neighbors), the structure absorbs the damage → **Benign**

3. **"Compensatory Mutation"**: If some pairs weaken (negative delta) but others strengthen (positive delta), the structure reorganizes but survives → likely **Benign**

4. **"Silent Mutation"**: If the SNP changes the base but doesn't affect ANY base pair probabilities (all deltas ≈ 0), the structure is unchanged → **Benign**
