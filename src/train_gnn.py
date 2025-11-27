import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling
from build_graph import load_graph, load_mappings
import os

# GNN composed of a simple GCN and GAT, both 2 layers

class GCNsimple(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        # 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2
        x = self.conv2(x, edge_index)
        return x
    
    def encode(self, x, edge_index):
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)


class GATsimple(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
    
    def forward(self, x, edge_index):
        # 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2
        x = self.conv2(x, edge_index)
        return x
    
    def encode(self, x, edge_index):
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)

# splits edges into train/val/test index sets
def split_edges(data, train_ratio=0.85, val_ratio=0.10):

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

# train one epoch per prediction link
def train_link_prediction(model, data, train_edge_index, optimizer):

    model.train()
    optimizer.zero_grad()
    
    z = model.encode(data.x, train_edge_index)
    
    pos_edge_index = train_edge_index
    
    # negative sampling for some reason
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.shape[1]
    )
    
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # binary cross entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# tests GNN with loaded graph, returns as AUC score
@torch.no_grad()
def test_link_prediction(model, data, edge_index):
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    
    z = model.encode(data.x, data.edge_index)
    
    pos_pred = model.decode(z, edge_index).sigmoid()
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=edge_index.shape[1]
    )
    neg_pred = model.decode(z, neg_edge_index).sigmoid()
    
    # compute AUC
    pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).cpu().numpy()
    
    auc = roc_auc_score(labels, pred)
    return auc


def save_model(model, optimizer, epoch, loss, filepath='./saves/model_checkpoint.pt'):
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


def load_model(filepath='./saves/model_checkpoint.pt', device='cpu'):

    checkpoint = torch.load(filepath, map_location=device)
    
    # recreate model
    config = checkpoint['model_config']
    if config['type'] == 'SimpleGCN':
        model = GCNsimple(
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim']
        )
    elif config['type'] == 'SimpleGAT':
        model = GATsimple(
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model loaded from {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return model, optimizer, epoch, loss


def create_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    return embeddings

# using cosine similarity, finds most similar nodes
# returns a list of (node, similarity) tuples
def find_similar_nodes(node_name, embeddings, mappings, top_k=5):
    node_to_idx = mappings['node_to_idx']
    idx_to_node = mappings['idx_to_node']
    
    if node_name not in node_to_idx:
        print(f"Node '{node_name}' not found in graph")
        return []
    
    query_idx = node_to_idx[node_name]
    query_emb = embeddings[query_idx]
    
    similarities = F.cosine_similarity(query_emb.unsqueeze(0), embeddings)
    
    # get top-k most similar
    top_scores, top_indices = similarities.topk(top_k + 1)      # +1 to exclude self
    
    results = []
    for score, idx in zip(top_scores[1:], top_indices[1:]):     # skip self
        results.append((idx_to_node[idx.item()], score.item()))
    
    return results
