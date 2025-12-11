import re
from pathlib import Path
from typing import List, Dict

# Parse input concept profiles fron conceptnet

def parse_file(filepath: str) -> Dict:
    
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    # main node pattern matching
    node_match = re.search(r'\*\*Node:\*\* `(.+?)` → \*\*(.+?)\*\*', text)
    main_node = node_match.group(2) if node_match else None

    # edge pattern matching idk it works
    edge_pattern = re.compile(
        r'- (?:\*\*)?(.+?)(?:\*\*)? (\w+) → (?:\*)?(?:\*\*)?(.+?)(?:\*\*)?(?:\*)? `\[([0-9.]+)\]`'
    )

    edges = []
    for match in edge_pattern.finditer(text):
        source = match.group(1).strip().strip('*')
        relation = match.group(2).strip()
        target = match.group(3).strip().strip('*')
        weight = float(match.group(4))
        
        edges.append({
            'source': source,
            'relation': relation,
            'target': target,
            'weight': weight
        })
    
    return {
        'node': main_node,
        'edges': edges
    }

def parse_directory(directory: str) -> List[Dict]:
    directory_path = Path(directory)
    profiles = []

    for filepath in directory_path.glob("*.txt"):
        profile = parse_file(filepath)
        profiles.append(profile)

    return profiles

def extract_all_nodes(profiles: List[Dict]) -> List[str]:
    nodes = set()
    
    for profile in profiles:

        if profile['node']:
            nodes.add(profile['node'])
        
        # add all source and target nodes from edges
        for edge in profile['edges']:
            nodes.add(edge['source'])
            nodes.add(edge['target'])
    
    return sorted(list(nodes))


def extract_all_relations(profiles: List[Dict]) -> List[str]:
    relations = set()
    
    for profile in profiles:
        for edge in profile['edges']:
            relations.add(edge['relation'])
    
    return sorted(list(relations))


def extract_all_edges(profiles: List[Dict]) -> List[Dict]:

    all_edges = []
    
    for profile in profiles:
        all_edges.extend(profile['edges'])
    
    return all_edges


