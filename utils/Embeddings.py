import requests
import numpy as np
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class EmbeddingModel(Embeddings):
    def __init__(self, mode=1):
        self.url = 'http://10.100.1.54:15001/embedding'
        self._dimension = 1792
        self.mode = mode
        if mode == 2:
            self.model = SentenceTransformer(
                model_name_or_path="BAAI/bge-large-en-v1.5",
                cache_folder='./models',
            )

    def embed_query(self, texts: str) -> List[float]:
        if self.mode == 1:
            embedding = self.model.encode([texts])
            np.array(embedding.astype(np.float64)[0]).tolist()
        
    def embed_query(self, texts: str) -> list[float]:
        if self.mode == 1:
            response = requests.post(self.url, json={"query": texts })
            res = response.json()["embeddings"]
            return np.array(res).astype(np.float64)
        else:
            embedding = self.model.encode([texts])
            res = np.array(embedding.astype(np.float64)[0]).tolist()
            return np.array(res).astype(np.float64)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            results.append(self.embed_query(t).tolist())
        return results
    
    def to_embeddings(self, data, **kwargs):
        if self.mode == 1:
            response = requests.post(self.url, json={"query": data })
            res = response.json()["embeddings"]
            return np.array(res).astype("float32")
        else:
            embedding = self.model.encode([data])
            return np.array(embedding.astype(np.float64)[0]).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension