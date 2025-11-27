import torch
from torch_geometric.data import Data
from typing import List, Dict
import pickle

# Building graph from conceptnet profiles:

# 1. create node mapping and relation mapping
# 2. map them to indecies
# 3. create edge index from profiles
# 4. convert to tensors and create node features
# 5. create PyG object

def create_node_mapping(nodes: List[str]) -> Dict:

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    return {
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node
    }

def create_relation_mapping(relations: List[str]) -> Dict:
    
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}
    idx_to_relation = {idx: rel for rel, idx in relation_to_idx.items()}
    
    return {
        'relation_to_idx': relation_to_idx,
        'idx_to_relation': idx_to_relation
    }

def build_edge_index(edges: List[Dict], node_to_idx: Dict, relation_to_idx: Dict):

    edge_list = []
    edge_types = []
    edge_weights = []
    
    for edge in edges:

        source_idx = node_to_idx[edge['source']]
        target_idx = node_to_idx[edge['target']]
        relation_idx = relation_to_idx[edge['relation']]
        weight = edge['weight']
        
        edge_list.append([source_idx, target_idx])
        edge_types.append(relation_idx)
        edge_weights.append(weight)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_type, edge_weight


def create_node_features(num_nodes: int, feature_dim: int = 128) -> torch.Tensor:
    return torch.randn(num_nodes, feature_dim)


def build_pyg_graph(nodes: List[str], 
                    relations: List[str], 
                    edges: List[Dict],
                    feature_dim: int = 128) -> Data:
    
    node_mapping = create_node_mapping(nodes)
    relation_mapping = create_relation_mapping(relations)
    
    node_to_idx = node_mapping['node_to_idx']
    relation_to_idx = relation_mapping['relation_to_idx']
    
    edge_index, edge_type, edge_weight = build_edge_index(
        edges, node_to_idx, relation_to_idx
    )
    
    x = create_node_features(len(nodes), feature_dim)
    
    data = Data(
        x=x,                        # Node features [num_nodes, feature_dim]
        edge_index=edge_index,      # Edge connections [2, num_edges]
        edge_type=edge_type,        # Relation type for each edge [num_edges]
        edge_weight=edge_weight,    # Weight for each edge [num_edges]
        num_nodes=len(nodes)
    )
    
    return data, node_mapping, relation_mapping


# pyG graph and mappings have to be saved separately

def save_graph(data: Data, filepath: str = 'graph.pt'):
    torch.save(data, filepath)
    print(f"Graph saved to {filepath}")


def save_mappings(node_mapping: Dict, relation_mapping: Dict, 
                  filepath: str = 'mappings.pkl'):
    mappings = {
        **node_mapping,
        **relation_mapping
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(mappings, f)
    
    print(f"Mappings saved to {filepath}")


def load_graph(filepath: str = 'graph.pt') -> Data:
    return torch.load(filepath)

def load_mappings(filepath: str = 'mappings.pkl') -> Dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)