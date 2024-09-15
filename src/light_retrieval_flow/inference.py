import os

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

root_data_dir = os.environ['DATA_DIR']
model_dir = os.path.join(root_data_dir, 'pipelines_data', 'models')

# Load the ONNX model
onnx_model_path = os.path.join(model_dir, "model.onnx")
model_name = 'multi-qa-distilbert-cos-v1'
ort_session = ort.InferenceSession(onnx_model_path)

# Load the tokenizer
# Replace 'your-model-name' with the name of the original model you used
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))
original_model = AutoModel.from_pretrained(os.path.join(model_dir, model_name))

def get_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Use the original model to get embeddings
    with torch.no_grad():
        outputs = original_model(**inputs)
    
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).numpy()
    
    return embedding

# Example usage
sentences = [
    "This is an example sentence.",
    "Another sentence for embedding.",
]

for sentence in sentences:
    embedding = get_embedding(sentence)
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 values): {embedding[0][:5]}")
    print()

# Example of comparing two sentences
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

emb1 = get_embedding(sentences[0]).flatten()
emb2 = get_embedding(sentences[1]).flatten()

similarity = cosine_similarity(emb1, emb2)
print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}': {similarity}")