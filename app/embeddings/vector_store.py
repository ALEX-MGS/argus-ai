import faiss
import numpy as np

class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, vector, text):
        vector = np.array([vector]).astype("float32")
        self.index.add(vector)
        self.texts.append(text)
    def save(self, path="faiss.index"):
        faiss.write_index(self.index, path)

    def load(self, path="faiss.index"):
        self.index = faiss.read_index(path)

    def search(self, query_vector, k=3, threshold=1.0):
        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []
        
    
        for distance, idx in zip(distances[0], indices[0]):
            if distance < threshold:
                results.append(self.texts[idx])
                return results
    


        
