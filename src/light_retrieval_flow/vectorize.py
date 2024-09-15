import json
import os
import numpy as np

from light_retrieval_flow.utils import clean_text, read_csv_as_dicts


def get_pytorch_model(models_dir, model_name='multi-qa-distilbert-cos-v1'):
  from sentence_transformers import SentenceTransformer

  model_path = os.path.join(models_dir, model_name)

  if not os.path.exists(model_path):
      print('huggingface model loading...')
      embedder = SentenceTransformer(model_name)
      embedder.save(model_path)
  else:
      print('pretrained model loading...')
      embedder = SentenceTransformer(model_name_or_path=model_path)
  print('model loadind done')

  return embedder

embedder = None
def get_or_create_embedder(models_dir, model_name):
    global embedder
    if embedder is None:
        embedder = get_pytorch_model(models_dir, model_name)
    return embedder

def normed_vector(v):
    norm = np.sqrt((v * v).sum())
    v_norm = v / norm
    return v_norm

def prepare_catalog(docs):
    index = []
    corpus = []
    for doc in docs:
        index.append(doc['doc_id'])
        corpus.append(clean_text(doc['content']))
    return index, corpus

def download_huggingface_model():
    root_dir = os.environ['DATA_DIR']
    models_dir=os.path.join(root_dir, 'pipelines-data', 'models')
    get_pytorch_model(models_dir, model_name='multi-qa-distilbert-cos-v1')

def eval_embeds(root_dir, documents):
    output_path = os.path.join(root_dir, 'pipelines-data', 'models', 'embeds.npy')
    
    index, corpus = prepare_catalog(documents)
    print(f'Index len {len(index)}, Index len {len(corpus)}')
    models_dir=os.path.join(root_dir, 'pipelines-data', 'models')
    if not os.path.exists(output_path):
        embedder = get_or_create_embedder(models_dir, model_name='multi-qa-distilbert-cos-v1')
        embeds = embedder.encode(corpus, show_progress_bar=True)
        print(f'Embeds shape: {embeds.shape}')
        np.save(output_path, embeds)
    data_path = os.path.join(models_dir, 'embeds_index.json')
    with open(data_path, 'w') as f:
      json.dump(index, f)

class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [{'score': scores[i], 'doc': self.documents[i]} for i in idx]


def get_search_engine(root_dir):
    models_dir = os.path.join(root_dir, 'pipelines-data')

    index_file_path = os.path.join(models_dir, 'embeds_index.json')
    embeds_file_path = os.path.join(models_dir, 'embeds.npy')
    with open(index_file_path, 'r') as f:
        index = json.load(f)
    embeds = np.load(embeds_file_path)
    print(f'Data loaded: {embeds.shape}, {len(index)}')

    search_engine = VectorSearchEngine(documents=index, embeddings=embeds)
    # embedder = get_or_create_embedder(models_dir, model_name='multi-qa-distilbert-cos-v1')
    # v = embedder.encode(entry['query'])
    # search_result = search_engine.search(v, num_results=30)
    # return sum(res), len(res)

def export_onnx(models_dir):
    import onnxruntime
    import torch
    output_path = os.path.join(models_dir, 'model.onnx')
    model = get_or_create_embedder(models_dir, model_name='multi-qa-distilbert-cos-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    sentences = ["This is a sample sentence for ONNX conversion."]

    # Tokenize the sentence(s)
    inputs = model.tokenize(sentences)

    # Convert tokenized inputs to tensors
    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device)

    # Create a wrapper class to handle the forward pass
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model({'input_ids': input_ids, 'attention_mask': attention_mask})

    # Wrap the model
    wrapped_model = ModelWrapper(model)

    # Export the model to ONNX
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['sentence_embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'sentence_embedding': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f'Model saved to {output_path}')

if __name__ == '__main__':
    root_data_dir = os.environ['DATA_DIR']
    csv_data_path = os.path.join(root_data_dir, 'pipelines_data', 'knowledgebase.csv')
    index_entries = read_csv_as_dicts(csv_data_path)
    # eval_embeds(root_data_dir, index_entries)
    export_onnx(os.path.join(root_data_dir, 'pipelines_data', 'models'))
