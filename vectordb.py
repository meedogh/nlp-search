import faiss
import numpy as np
import pandas as pd

class VectorDB:
    def __init__(self, dim, index_path="vector_index.faiss", meta_path="vector_metadata.csv"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = pd.DataFrame()
        self.is_loaded = False

    def build(self, df, embedding_col="embedding"):
        embeddings = np.array(df[embedding_col].to_list()).astype("float32")
        faiss.normalize_L2(embeddings)  

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        self.metadata = df.drop(columns=[embedding_col]).reset_index(drop=True)
        self.is_loaded = True

    def save(self):
        if not self.is_loaded:
            raise ValueError("No index to save. Build or load it first.")
        faiss.write_index(self.index, self.index_path)
        self.metadata.to_csv(self.meta_path, index=False)
        print(f"VectorDB saved: {self.index_path} + {self.meta_path}")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        self.metadata = pd.read_csv(self.meta_path)
        self.is_loaded = True
        print(f"VectorDB loaded: {self.index_path} + {self.meta_path}")

    def search(self, query_vec, top_k=5):
        if not self.is_loaded:
            raise ValueError("Load or build the index first.")

        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k)
        results = self.metadata.iloc[indices[0]].copy()
        results["similarity"] = distances[0]
        return results.reset_index(drop=True)

    def add(self, df_new, embedding_col="embedding"):
        new_embeddings = np.array(df_new[embedding_col].to_list()).astype("float32")
        faiss.normalize_L2(new_embeddings)

        self.index.add(new_embeddings)
        self.metadata = pd.concat([self.metadata, df_new.drop(columns=[embedding_col])], ignore_index=True)
        print(f"Added {len(df_new)} new documents.")
