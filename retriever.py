import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.embeddings = np.load("data/embeddings.npy")

        with open("data/protocols.pkl", "rb") as f:
            self.protocols = pickle.load(f)

    def search(self, query, top_k=10):
        query_vec = self.model.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]

        results = [self.protocols[i] for i in top_indices]

        return results  