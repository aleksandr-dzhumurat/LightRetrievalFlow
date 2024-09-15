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

if __name__ == '__main__':
    root_data_dir = os.environ['DATA_DIR']
    csv_data_path = os.path.join(root_data_dir, 'pipelines_data', 'knowledgebase.csv')
    index_entries = read_csv_as_dicts(csv_data_path)
    eval_embeds(root_data_dir, index_entries)
