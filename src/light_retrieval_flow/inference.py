import os

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class ONNXVectorizer:
    def __init__(self, model_dir, model_name):
        self.model = AutoModel.from_pretrained(os.path.join(model_dir, model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))

    def get_embedding(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).numpy()
        
        return embedding

def get_embedder():
    root_data_dir = os.environ['DATA_DIR']
    model_dir = os.path.join(root_data_dir, 'pipelines_data', 'models')
    model_name = 'multi-qa-distilbert-cos-v1'
    onnx_vectorizer = ONNXVectorizer(model_dir, model_name)

    # Load the ONNX model
    # onnx_model_path = os.path.join(model_dir, "model.onnx")
    # ort_session = ort.InferenceSession(onnx_model_path)
    return onnx_vectorizer

onnx_vectorizer = get_embedder()

if __name__ == '__main__':
    sentences = [
        "This is an example sentence.",
        "Another sentence for embedding.",
    ]

    for sentence in sentences:
        embedding = onnx_vectorizer.get_embedding(sentence)
        print(f"Sentence: {sentence}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 values): {embedding[0][:5]}")
        print()


    emb1 = onnx_vectorizer.get_embedding(sentences[0]).flatten()
    emb2 = onnx_vectorizer.get_embedding(sentences[1]).flatten()

    similarity = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}': {similarity}")