# CommonsenseQA-AI

Answer common sense questions using ConceptNet and PyTorch Geometric GNNs.

## Quick Start

```bash
# Install dependencies
pip install torch torch-geometric

# Build graph from ConceptNet profiles
python build_graph.py

# Train GNN model
python train_gnn.py

# Ask questions
python main.py
```

## Example

```
Question: What is a dog?
Answer: A dog is a type of pet and mammal. It is loyal and friendly. Dogs can bark and run.

Question: Where do you find cats?
Answer: You can typically find a cat at home, house, or outside.
```

## How It Works

1. **Parse** ConceptNet semantic profiles → nodes and edges
2. **Build** PyTorch Geometric graph (988 nodes, 1.3k edges)
3. **Train** 2-layer GCN on link prediction task
4. **Answer** questions using learned embeddings + graph structure

## Files

- `parse_profile.py` - Parse ConceptNet text files
- `build_graph.py` - Build PyTorch Geometric graph
- `train_gnn.py` - GNN training and model management
- `main.py` - Question answering interface

## Architecture

```
ConceptNet Profiles → Graph → GCN → Embeddings → Q&A System
                      (988 nodes)  (32-dim)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- ConceptNet semantic profile text files in `./conceptnet_profiles/`

## License

MIT