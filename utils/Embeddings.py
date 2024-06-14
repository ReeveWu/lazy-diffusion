import requests
import numpy as np
from typing import List, Optional
from langchain_core.embeddings import Embeddings

class EmbeddingModel(Embeddings):
    def __init__(self):
        self.url = 'http://10.100.1.54:15001/embedding'
        self._dimension = 1792
        
    def embed_query(self, texts: str) -> list[float]:
        response = requests.post(self.url, json={"query": texts })
        res = response.json()["embeddings"]

        return np.array(res).astype(np.float64)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            results.append(self.embed_query(t).tolist())
        return results
    
    def to_embeddings(self, data, **kwargs):
        response = requests.post(self.url, json={"query": data })
        res = response.json()["embeddings"]
        return np.array(res).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension