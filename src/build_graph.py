"""
build_graph.py
Build PyTorch Geometric graph from parsed ConceptNet data
"""

import torch
from torch_geometric.data import Data
from typing import List, Dict
import pickle


def create_node_mapping(nodes: List[str]) -> Dict:
    """
    Create mapping from node names to indices
    
    Args:
        nodes: List of unique node names
    
    Returns:
        Dict with 'node_to_idx' and 'idx_to_node'
    """
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    return {
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node
    }


def create_relation_mapping(relations: List[str]) -> Dict:
    """
    Create mapping from relation types to indices
    
    Args:
        relations: List of unique relation types
    
    Returns:
        Dict with 'relation_to_idx' and 'idx_to_relation'
    """
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}
    idx_to_relation = {idx: rel for rel, idx in relation_to_idx.items()}
    
    return {
        'relation_to_idx': relation_to_idx,
        'idx_to_relation': idx_to_relation
    }


def build_edge_index(edges: List[Dict], node_to_idx: Dict, relation_to_idx: Dict):
    """
    Build edge index and attributes for PyTorch Geometric
    
    Args:
        edges: List of edge dicts with source, target, relation, weight
        node_to_idx: Mapping from node names to indices
        relation_to_idx: Mapping from relation names to indices
    
    Returns:
        edge_index: [2, num_edges] tensor
        edge_type: [num_edges] tensor of relation indices
        edge_weight: [num_edges] tensor of weights
    """
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
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_type, edge_weight


def create_node_features(num_nodes: int, feature_dim: int = 128) -> torch.Tensor:
    """
    Create initial node features (random initialization)
    
    Args:
        num_nodes: Number of nodes
        feature_dim: Dimension of features
    
    Returns:
        Node feature matrix [num_nodes, feature_dim]
    """
    return torch.randn(num_nodes, feature_dim)


def build_pyg_graph(nodes: List[str], 
                    relations: List[str], 
                    edges: List[Dict],
                    feature_dim: int = 128) -> Data:
    """
    Build complete PyTorch Geometric Data object
    
    Args:
        nodes: List of unique node names
        relations: List of unique relation types
        edges: List of edge dicts
        feature_dim: Dimension of node features
    
    Returns:
        PyTorch Geometric Data object
    """
    # Create mappings
    node_mapping = create_node_mapping(nodes)
    relation_mapping = create_relation_mapping(relations)
    
    node_to_idx = node_mapping['node_to_idx']
    relation_to_idx = relation_mapping['relation_to_idx']
    
    # Build edge tensors
    edge_index, edge_type, edge_weight = build_edge_index(
        edges, node_to_idx, relation_to_idx
    )
    
    # Create node features
    x = create_node_features(len(nodes), feature_dim)
    
    # Create PyG Data object
    data = Data(
        x=x,                    # Node features [num_nodes, feature_dim]
        edge_index=edge_index,  # Edge connections [2, num_edges]
        edge_type=edge_type,    # Relation type for each edge [num_edges]
        edge_weight=edge_weight,# Weight for each edge [num_edges]
        num_nodes=len(nodes)
    )
    
    return data, node_mapping, relation_mapping


def save_graph(data: Data, filepath: str = 'graph.pt'):
    """Save PyTorch Geometric graph"""
    torch.save(data, filepath)
    print(f"Graph saved to {filepath}")


def save_mappings(node_mapping: Dict, relation_mapping: Dict, 
                  filepath: str = 'mappings.pkl'):
    """Save node and relation mappings"""
    mappings = {
        **node_mapping,
        **relation_mapping
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(mappings, f)
    
    print(f"Mappings saved to {filepath}")


def load_graph(filepath: str = 'graph.pt') -> Data:
    """Load PyTorch Geometric graph"""
    return torch.load(filepath, weights_only=False)


def load_mappings(filepath: str = 'mappings.pkl') -> Dict:
    """Load node and relation mappings"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)