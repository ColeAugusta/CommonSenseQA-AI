"""
train_gnn.py
Complete GNN training with model saving
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling
from build_graph import load_graph, load_mappings
import os


class GCNsimple(torch.nn.Module):
    """Simple 2-layer Graph Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        return x
    
    def encode(self, x, edge_index):
        """Get node embeddings"""
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_index):
        """Decode embeddings to predict edge existence"""
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)


class SimpleGAT(torch.nn.Module):
    """Simple 2-layer Graph Attention Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
    
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        return x
    
    def encode(self, x, edge_index):
        """Get node embeddings"""
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_index):
        """Decode embeddings to predict edge existence"""
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)


def split_edges(data, train_ratio=0.85, val_ratio=0.10):
    """
    Split edges into train/val/test sets
    
    Args:
        data: PyTorch Geometric Data object
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        train_edge_index, val_edge_index, test_edge_index
    """
    num_edges = data.edge_index.shape[1]
    perm = torch.randperm(num_edges)
    
    train_size = int(train_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    
    train_edge_index = data.edge_index[:, train_idx]
    val_edge_index = data.edge_index[:, val_idx]
    test_edge_index = data.edge_index[:, test_idx]
    
    return train_edge_index, val_edge_index, test_edge_index


def train_link_prediction(model, data, train_edge_index, optimizer):
    """
    Train one epoch for link prediction
    
    Args:
        model: GNN model
        data: Graph data
        train_edge_index: Training edges
        optimizer: Optimizer
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    optimizer.zero_grad()
    
    # Encode node embeddings
    z = model.encode(data.x, train_edge_index)
    
    # Positive edges
    pos_edge_index = train_edge_index
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.shape[1]
    )
    
    # Decode
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Binary cross entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def test_link_prediction(model, data, edge_index):
    """
    Test link prediction performance
    
    Args:
        model: GNN model
        data: Graph data
        edge_index: Test edges
        
    Returns:
        AUC score
    """
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    
    # Encode
    z = model.encode(data.x, data.edge_index)
    
    # Positive edges
    pos_pred = model.decode(z, edge_index).sigmoid()
    
    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=edge_index.shape[1]
    )
    neg_pred = model.decode(z, neg_edge_index).sigmoid()
    
    # Compute AUC
    pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).cpu().numpy()
    
    auc = roc_auc_score(labels, pred)
    return auc


def save_model(model, optimizer, epoch, loss, filepath='./saves/model_checkpoint.pt'):
    """
    Save model checkpoint
    
    Args:
        model: GNN model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'type': model.__class__.__name__,
            'input_dim': model.conv1.in_channels,
            'hidden_dim': model.conv1.out_channels,
            'output_dim': model.conv2.out_channels
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='./saves/model_checkpoint.pt', device='cpu'):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        device: Device to load model on
        
    Returns:
        model, optimizer, epoch, loss
    """
    # Load with weights_only=False for PyTorch 2.6+
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Recreate model
    config = checkpoint['model_config']
    if config['type'] == 'GCNsimple':
        model = GCNsimple(
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim']
        )
    elif config['type'] == 'SimpleGAT':
        model = SimpleGAT(
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Recreate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model loaded from {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return model, optimizer, epoch, loss


def create_node_embeddings(model, data):
    """
    Generate embeddings for all nodes
    
    Args:
        data: PyTorch Geometric Data object
        model: Trained GNN model
    
    Returns:
        Node embeddings tensor
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    return embeddings


def find_similar_nodes(node_name, embeddings, mappings, top_k=5):
    """
    Find most similar nodes using cosine similarity
    
    Args:
        node_name: Name of the query node
        embeddings: Node embedding matrix
        mappings: Dict with node mappings
        top_k: Number of similar nodes to return
    
    Returns:
        List of (node_name, similarity_score) tuples
    """
    node_to_idx = mappings['node_to_idx']
    idx_to_node = mappings['idx_to_node']
    
    if node_name not in node_to_idx:
        print(f"Node '{node_name}' not found in graph")
        return []
    
    # Get query node embedding
    query_idx = node_to_idx[node_name]
    query_emb = embeddings[query_idx]
    
    # Compute cosine similarity with all nodes
    similarities = F.cosine_similarity(query_emb.unsqueeze(0), embeddings)
    
    # Get top-k most similar
    top_scores, top_indices = similarities.topk(top_k + 1)  # +1 to exclude self
    
    results = []
    for score, idx in zip(top_scores[1:], top_indices[1:]):  # Skip self
        results.append((idx_to_node[idx.item()], score.item()))
    
    return results


def main():
    """Complete training pipeline"""
    
    # Load graph
    print("="*60)
    print("Loading graph...")
    print("="*60)
    data = load_graph('./saves/conceptnet_graph.pt')
    mappings = load_mappings('./saves/conceptnet_mappings.pkl')
    
    print(f"Graph loaded:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Feature dim: {data.x.shape[1]}")
    
    # Split edges
    print("\n" + "="*60)
    print("Splitting edges...")
    print("="*60)
    train_edge_index, val_edge_index, test_edge_index = split_edges(data)
    print(f"Train edges: {train_edge_index.shape[1]}")
    print(f"Val edges: {val_edge_index.shape[1]}")
    print(f"Test edges: {test_edge_index.shape[1]}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    input_dim = data.x.shape[1]
    hidden_dim = 64
    output_dim = 32
    
    model = GCNsimple(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    num_epochs = 100
    best_val_auc = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_link_prediction(model, data, train_edge_index, optimizer)
        
        # Validate every 10 epochs
        if epoch % 10 == 0:
            val_auc = test_link_prediction(model, data, val_edge_index)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_model(model, optimizer, epoch, loss, './saves/best_model.pt')
    
    # Final test
    print("\n" + "="*60)
    print("Testing...")
    print("="*60)
    test_auc = test_link_prediction(model, data, test_edge_index)
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save final model
    save_model(model, optimizer, num_epochs, loss, './saves/final_model.pt')
    
    # Generate embeddings
    print("\n" + "="*60)
    print("Generating embeddings...")
    print("="*60)
    embeddings = create_node_embeddings(model, data)
    torch.save(embeddings, './saves/node_embeddings.pt')
    print(f"Embeddings saved: {embeddings.shape}")
    
    # Test similarity search
    print("\n" + "="*60)
    print("Testing similarity search...")
    print("="*60)
    
    test_nodes = ['dog', 'cat', 'computer']
    for node in test_nodes:
        if node in mappings['node_to_idx']:
            print(f"\nMost similar to '{node}':")
            similar = find_similar_nodes(node, embeddings, mappings, top_k=5)
            for i, (similar_node, score) in enumerate(similar, 1):
                print(f"  {i}. {similar_node} (similarity: {score:.4f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nSaved files:")
    print("  - best_model.pt (best validation model)")
    print("  - final_model.pt (final epoch model)")
    print("  - node_embeddings.pt (node embeddings)")

if __name__ == "__main__":
    main()