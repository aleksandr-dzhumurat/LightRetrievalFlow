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
