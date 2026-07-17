import faiss
import numpy as np
import json

class VectorStore:

    def __init__(self, dimension: int = 1536):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, vector, text, source ="unknown"):
        vector = np.array([vector]).astype("float32")
        self.index.add(vector)
        self.documents.append({"text": text,"source": source})

    def save(self, path="app/embeddings/faiss.index"):
        faiss.write_index(self.index, path)


        with open("app/embeddings/documents.json", "w") as f:
            json.dump(self.documents, f)
            


    def load(self, path="app/embeddings/faiss.index"):
        self.index = faiss.read_index(path)

        with open("app/embeddings/documents.json", "r") as f:
            self.documents = json.load(f)

    def search(self, query_vector, k=10, threshold=None):
        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if threshold is not None and distance > threshold:
                continue
            results.append(self.documents[idx])

        return results