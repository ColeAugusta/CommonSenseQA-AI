import torch
import torch.nn.functional as F
from train_gnn import (
    GCNsimple, 
    load_model, 
    create_node_embeddings,
)
from build_graph import load_graph, load_mappings
import re
import os

# Source of main for the entire CommonsenseQA-AI project

class CommonSenseQA:
    
    def __init__(self):
        self.model = None
        self.data = None
        self.mappings = None
        self.embeddings = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialize()
    
    def _initialize(self):
        print("Loading ConceptNet model...")
        
        self.data = load_graph('./saves/conceptnet_graph.pt')
        self.mappings = load_mappings('./saves/conceptnet_mappings.pkl')
        
        if os.path.exists('./saves/best_model.pt'):
            self.model, _, _, _ = load_model('./saves/best_model.pt', self.device)
        else:
            # creates untrained model if model doesnt exist. shouldnt use this
            input_dim = self.data.x.shape[1]
            self.model = GCNsimple(input_dim, 64, 32)
            print("Warning: Using untrained model")
        
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        self.embeddings = create_node_embeddings(self.model, self.data)
        
        print("Model ready.\n")
    
    # simple pulls concepts from question
    def extract_concepts(self, text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        concepts = [w for w in words if w in self.mappings['node_to_idx']]
        
        return concepts
    
    def get_concept_embedding(self, concept):
        if concept not in self.mappings['node_to_idx']:
            return None
        
        idx = self.mappings['node_to_idx'][concept]
        return self.embeddings[idx]
    
    def find_related_concepts(self, concept, top_k=5):
        emb = self.get_concept_embedding(concept)
        if emb is None:
            return []
        
        # cosine similarity to all concepts
        similarities = F.cosine_similarity(emb.unsqueeze(0), self.embeddings)
        
        # get top-K excluding self
        top_scores, top_indices = similarities.topk(top_k + 1)
        
        results = []
        for score, idx in zip(top_scores[1:], top_indices[1:]):
            node = self.mappings['idx_to_node'][idx.item()]
            results.append((node, score.item()))
        
        return results
    
    def get_edge_relations(self, concept):
        if concept not in self.mappings['node_to_idx']:
            return []
        
        node_idx = self.mappings['node_to_idx'][concept]
        
        # find edges where this node is the source
        mask = self.data.edge_index[0] == node_idx
        target_indices = self.data.edge_index[1][mask]
        edge_types = self.data.edge_type[mask]
        edge_weights = self.data.edge_weight[mask]
        
        relations = []
        for target_idx, edge_type, weight in zip(target_indices, edge_types, edge_weights):
            target_node = self.mappings['idx_to_node'][target_idx.item()]
            relation_type = self.mappings['idx_to_relation'][edge_type.item()]
            relations.append((relation_type, target_node, weight.item()))
        
        # sort by weight
        relations.sort(key=lambda x: x[2], reverse=True)
        
        return relations[:10]
    
    def answer_question(self, question):
        
        concepts = self.extract_concepts(question)
        
        if not concepts:
            return "I don't recognize any concepts from the question in my knowledge base."
        
        question_lower = question.lower()
        
        main_concept = concepts[0]
        
        # get direct relations for main concept
        relations = self.get_edge_relations(main_concept)
        relation_dict = {}
        for rel_type, target, weight in relations:
            if rel_type not in relation_dict:
                relation_dict[rel_type] = []
            relation_dict[rel_type].append(target)
        
        # generate NLP answer here based on question type
        # simple, template based
        
        # "What is" questions
        if any(phrase in question_lower for phrase in ['what is', 'what are', 'what\'s']):
            if 'IsA' in relation_dict:
                types = relation_dict['IsA'][:2]
                answer = f"A {main_concept} is a type of {' and '.join(types)}."
                
                # Add properties if available
                if 'HasProperty' in relation_dict:
                    props = relation_dict['HasProperty'][:2]
                    answer += f" It is {' and '.join(props)}."
                
                # Add capabilities
                if 'CapableOf' in relation_dict:
                    actions = relation_dict['CapableOf'][:2]
                    answer += f" {main_concept.capitalize()}s can {' and '.join(actions)}."
                
                return answer
            else:
                return f"A {main_concept} is a concept I know about, but I don't have a specific definition."
        
        # "Where" questions
        elif 'where' in question_lower:
            if 'AtLocation' in relation_dict:
                locations = relation_dict['AtLocation'][:3]
                if len(locations) == 1:
                    return f"You can typically find a {main_concept} at {locations[0]}."
                elif len(locations) == 2:
                    return f"You can typically find a {main_concept} at {locations[0]} or {locations[1]}."
                else:
                    return f"You can typically find a {main_concept} at {', '.join(locations[:-1])}, or {locations[-1]}."
            else:
                return f"I don't have specific location information about {main_concept}."
        
        # "What can/does" questions - capabilities
        elif any(phrase in question_lower for phrase in ['what can', 'what does', 'can it']):
            if 'CapableOf' in relation_dict:
                actions = relation_dict['CapableOf'][:3]
                if len(actions) == 1:
                    return f"A {main_concept} can {actions[0]}."
                elif len(actions) == 2:
                    return f"A {main_concept} can {actions[0]} and {actions[1]}."
                else:
                    return f"A {main_concept} can {', '.join(actions[:-1])}, and {actions[-1]}."
            else:
                return f"I don't have specific capability information about {main_concept}."
        
        # "Why" or "What for" questions - purpose
        elif 'why' in question_lower or 'what for' in question_lower or 'used for' in question_lower:
            if 'UsedFor' in relation_dict:
                uses = relation_dict['UsedFor'][:3]
                if len(uses) == 1:
                    return f"A {main_concept} is used for {uses[0]}."
                else:
                    return f"A {main_concept} is used for {', '.join(uses[:-1])}, and {uses[-1]}."
            else:
                return f"I don't have specific usage information about {main_concept}."
        
        # "What has" or "What does it have" questions - parts
        elif any(phrase in question_lower for phrase in ['what has', 'what does it have', 'parts of']):
            if 'HasA' in relation_dict:
                parts = relation_dict['HasA'][:3]
                if len(parts) == 1:
                    return f"A {main_concept} has {parts[0]}."
                else:
                    return f"A {main_concept} has {', '.join(parts[:-1])}, and {parts[-1]}."
            else:
                return f"I don't have specific part information about {main_concept}."
        
        # "How" questions
        elif 'how' in question_lower:
            if 'CapableOf' in relation_dict:
                actions = relation_dict['CapableOf'][:2]
                return f"A {main_concept} can {' and '.join(actions)}."
            else:
                return f"I don't have specific information about how a {main_concept} works."
        
        # Comparison questions (contains two concepts)
        elif len(concepts) >= 2:
            concept1, concept2 = concepts[0], concepts[1]
            
            # get similar concepts for both
            similar1 = self.find_related_concepts(concept1, top_k=5)
            similar2 = self.find_related_concepts(concept2, top_k=5)
            
            # check if they're in each other's similar list
            similar1_names = [name for name, _ in similar1]
            similar2_names = [name for name, _ in similar2]
            
            if concept2 in similar1_names or concept1 in similar2_names:
                return f"{concept1.capitalize()} and {concept2} are related concepts that share similar properties."
            else:
                # compare embeddings
                emb1 = self.get_concept_embedding(concept1)
                emb2 = self.get_concept_embedding(concept2)
                if emb1 is not None and emb2 is not None:
                    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    if sim > 0.7:
                        return f"{concept1.capitalize()} and {concept2} are quite similar concepts."
                    elif sim > 0.4:
                        return f"{concept1.capitalize()} and {concept2} are somewhat related."
                    else:
                        return f"{concept1.capitalize()} and {concept2} are not closely related concepts."
        
        # Default: provide general information
        # shouldn't be used, except for questions outside one of the link types
        else:
            answer_parts = []
            
            if 'IsA' in relation_dict:
                types = relation_dict['IsA'][:2]
                answer_parts.append(f"A {main_concept} is a {' and '.join(types)}.")
            
            if 'CapableOf' in relation_dict:
                actions = relation_dict['CapableOf'][:2]
                answer_parts.append(f"It can {' and '.join(actions)}.")
            
            if 'UsedFor' in relation_dict:
                uses = relation_dict['UsedFor'][:2]
                answer_parts.append(f"It is used for {' and '.join(uses)}.")
            
            if answer_parts:
                return " ".join(answer_parts)
            else:
                return f"I know about {main_concept}, but I don't have detailed information to answer your question."
    
    # main QA system loop
    def run(self):
        print("Common Sense Q&A System")
        print("Ask me anything! (type 'quit' to exit)")
        print("-" * 60)
        print()
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                print()
                answer = self.answer_question(question)
                print(answer)
                print()
                print("-" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()


def main():
    qa = CommonSenseQA()
    qa.run()


if __name__ == "__main__":
    main()