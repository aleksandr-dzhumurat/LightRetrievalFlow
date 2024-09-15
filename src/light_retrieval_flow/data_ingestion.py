import os
import re
import requests
import json

import yaml
import pandas as pd
from typing import List


from light_retrieval_flow.utils import clean_text, read_csv_as_dicts


ZINCSEARCH_URL = os.environ["ZINCSEARCH_URL"]
USERNAME=os.environ['ZINCSEARCH_USERNAME']
PASSWORD=os.environ['ZINCSEARCH_PASSWORD']
"""
    index_entries = read_csv_as_dicts(root_dir)
    load_bulk_documents(index_entries)
    res = search_documents('Lemon, headache')
"""

def load_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def create_index(index_config: dict):
    url = f"{ZINCSEARCH_URL}/api/index"
    headers = {"Content-Type": "application/json"}
    # index_config = {
    #     "name": index_name,
    #     "storage_type": "disk",
    #     "shard_num": 1,
    #     "mappings": {
    #         "properties": {
    #             "doc_id": {
    #                 "type": "text",
    #                 "index": True,
    #                 "store": True,
    #                 "highlightable": True
    #             },
    #             "content": {
    #                 "type": "text",
    #                 "index": True,
    #                 "store": True,
    #                 "highlightable": True
    #             },
    #             "category": {
    #                 "type": "keyword",
    #                 "index": True,
    #                 "sortable": True,
    #                 "aggregatable": True
    #             },
    #             "content_len": {
    #                 "type": "integer",
    #                 "index": True,
    #                 "sortable": True,
    #                 "aggregatable": False
    #             }
    #         }
    #     },
    #     "settings": {
    #         "analysis": {
    #             "analyzer": {
    #                 "default": {
    #                     "type": "standard"
    #                 }
    #             }
    #         }
    #     }
    # }
    response = requests.post(url, headers=headers, data=json.dumps(index_config), auth=(USERNAME, PASSWORD))
    if response.status_code == 200:
        print(f"Index {index_config['name']} created successfully.")
    else:
        print(f"Failed to create index: {response.status_code}, {response.text}")

def load_document(data, index_name):
    url = f"{ZINCSEARCH_URL}/api/{index_name}/_doc"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data), auth=(USERNAME, PASSWORD))
    if response.status_code == 200:
        pass
    else:
        print(f"Failed to load documents: {response.status_code}, {response.text}")

def load_bulk_documents(index_name, documents):
    for doc in documents:
        doc['content'] = clean_text(doc['content'])
        load_document(doc, index_name)
    print("Document loaded successfully.")

def search_documents(index_name, query, category, limit=10, syntax='elastic'):
    """
    Search documents in the ZincSearch index based on a query and category.

    results = search_documents('your_index', 'Lemon, headache', 'news')
    print(results)

    Args:
        index_name (str): The name of the ZincSearch index.
        query (str): The search query (e.g., 'Lemon, headache').
        category (str): The category to filter by (e.g., 'flower', 'news').
        limit (int): The maximum number of search results to return (default: 10).
    """
    fields_list = ['content']  # List of fields to search within
    headers = {"Content-Type": "application/json"}
    if syntax == 'zinc':
        url = f"{ZINCSEARCH_URL}/api/{index_name}/_search"
        search_query = {
            "search_type": "match",
            "query":
            {
                "term": query,
            },
            # "from": 0, # use together with max_results for paginated results.
            # "max_results": 20,
            # "_source": [] # Leave this as empty array to return all fields.
        }
    else:
        url = f"{ZINCSEARCH_URL}/es/{index_name}/_search"
        search_query = {
            "query": {
                "bool": { 
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": fields_list,
                                "type": "best_fields"
                            }
                        },
                    ],
                    "should": [],
                    "must_not": [],
                    "filter": [ 
                        { "term":  { "category": category }},
                    ]
                }
            },
            "size": 10
        }
    response = requests.post(url, headers=headers, data=json.dumps(search_query), auth=(USERNAME, PASSWORD))
    if response.status_code == 200:
        return response.json().get('hits', {}).get('hits', [])[:limit]
    else:
        print(f"Search failed: {response.status_code}, {response.text}")
        return []

def pretty(search_results):
    """
        print(pretty(res))
    """
    result_docs = []

    include_fields = ['content']
    for hit in search_results:
        result_docs.append({k: v for k, v in  hit['_source'].items() if k in include_fields})
    return result_docs

# with open(config_path, "w") as f:
#     yaml.dump(config, f, default_flow_style=False)