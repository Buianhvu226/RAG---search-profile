import faiss
import numpy as np

class VectorSearcher:
    def __init__(self, embeddings, df):
        self.embeddings = embeddings
        self.df = df  # Store the full dataframe
        
    def search(self, query_embedding, top_k=10):
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        
        # Calculate cosine similarity
        similarities = np.dot(norm_embeddings, query_embedding.T).flatten()
        
        # Get top matches with similarity > 0.4
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter valid indices and return results
        results = []
        for idx in top_indices:
            try:
                if similarities[idx] > 0.4 and idx < len(self.df):
                    results.append((self.df.iloc[idx], similarities[idx]))
            except:
                continue
                
        return results