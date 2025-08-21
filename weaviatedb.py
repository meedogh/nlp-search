import weaviate
import numpy as np

class WeaviateDB:
    def __init__(self, client: weaviate.Client, model, class_name="Document"):
        self.client = client
        self.model = model
        self.class_name = class_name

    def compute_embeddings(self, df):
        embeddings = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row['tags']} {row['content']}"
            emb = self.model.encode(text, convert_to_numpy=True)
            embeddings.append(emb.tolist()) 
        df["embedding"] = embeddings
        return df

    def insert_documents(self, df, batch_size=50):
        with self.client.batch(batch_size=batch_size) as batch:
            for idx, row in df.iterrows():
                try:
                    batch.add_data_object(
                        data_object={
                            "document_id": row["document_id"],
                            "title": row["title"],
                            "author": row["author"],
                            "created_date": row["created_date"],
                            "last_modified": row["last_modified"],
                            "category": row["category"],
                            "tags": row["tags"],
                            "content": row["content"],
                            "vector": row["embedding"] 
                        },
                        class_name=self.class_name
                    )
                except Exception as e:
                    print(f"Error inserting row {idx} (document_id={row['document_id']}): {e}")

    def query(self, text, top_k=5, metric="cosine", where_filter=None):
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector}

        query_obj = (
            self.client.query
            .get(
                self.class_name,
                ["document_id", "title", "author", "created_date", "last_modified", "category", "tags", "content"]
            )
            .with_near_vector(near_vector)
            .with_additional(["distance"])
            .with_limit(top_k)
        )

        if where_filter:
            query_obj = query_obj.with_where(where_filter)

        result = query_obj.do()
        
        class_results = result.get("data", {}).get("Get", {}).get(self.class_name)
        if not class_results:
            return []

        hits = []
        for item in class_results:
            hits.append({
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "author": item.get("author"),
                "created_date": item.get("created_date"),
                "last_modified": item.get("last_modified"),
                "category": item.get("category"),
                "tags": item.get("tags"),
                "content": item.get("content"),
                "distance": item["_additional"]["distance"]
            })
        return hits


    
    def vector_search(self, text, top_k=5, metric="cosine"):
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector, "certainty": 0.7} if metric == "cosine" else {"vector": vector}
        query = self.client.query.get(self.class_name, ["document_id", "title", "author", "created_date", "last_modified", "category", "tags", "content"]) \
                    .with_near_vector(near_vector) \
                    .with_limit(top_k)
        if metric in ["cosine", "dot", "euclidean"]:
            query._additional["distance"] = metric
        return query.do()

    def filter_search(self, where_filter, top_k=5):
        query = self.client.query.get(self.class_name, ["document_id", "title", "author", "created_date", "last_modified", "category", "tags", "content"]) \
                    .with_where(where_filter) \
                    .with_limit(top_k)
        return query.do()

    def hybrid_search(self, text, where_filter=None, top_k=5, metric="cosine"):
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector, "certainty": 0.7} if metric == "cosine" else {"vector": vector}
        query = self.client.query.get(self.class_name,["document_id", "title", "author", "created_date", "last_modified", "category", "tags", "content"]) \
                    .with_near_vector(near_vector) \
                    .with_limit(top_k)
        if where_filter:
            query = query.with_where(where_filter)
        if metric in ["cosine", "dot", "euclidean"]:
            query._additional["distance"] = metric
        return query.do()
