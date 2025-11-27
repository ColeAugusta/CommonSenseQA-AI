from gradio_client import Client
import json
import requests

def query_node(client, word):
    try:
        result = client.predict(
            word=word,
            api_name="/get_semantic_profile"
        )
        return result
    except Exception as e:
        return {"error:", str(e)}


# used this to query the huggingface space for concept profiles

concepts = ['dog', 'cat', 'car', 'book', 'water', 'food', 'house', 
            'tree', 'bird', 'fish', 'computer', 'phone', 'table', 'chair',
            'person', 'animal', 'plant', 'tool', 'machine', 'building']
client = Client("https://cstr-conceptnet-normalized.hf.space")
for i, concept in enumerate(concepts):
    file_name = f"concept{i}.txt"
    with open(file_name, "w") as file:
        file.write(query_node(client, concept))