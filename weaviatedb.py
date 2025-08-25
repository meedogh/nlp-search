import weaviate
import numpy as np

class WeaviateDB:
    def __init__(self, client: weaviate.Client, model):
        self.client = client
        self.model = model

    def _row_to_text(self, row: dict, class_name: str) -> str:
        parts = [f"{class_name}:"]
        for key, value in row.items():
            if key == "embedding":
                continue
            if value is None:
                continue
            if isinstance(value, list):
                val_str = ", ".join(map(str, value))
            else:
                val_str = str(value)
            parts.append(f"{key}: {val_str}")
        return " | ".join(parts)

    def compute_embeddings(self, df, class_name: str):
        embeddings = []
        for _, row in df.iterrows():
            text_repr = self._row_to_text(row, class_name)
            emb = self.model.encode(text_repr, convert_to_numpy=True)
            embeddings.append(emb.tolist())
        df["embedding"] = embeddings
        return df

    def insert_objects(self, df, class_name: str, batch_size=50):
        with self.client.batch(batch_size=batch_size) as batch:
            for idx, row in df.iterrows():
                try:
                    data_obj = row.to_dict()
                    if "embedding" in data_obj:
                        vec = data_obj.pop("embedding")
                    else:
                        vec = None
                    batch.add_data_object(
                        data_object=data_obj,
                        class_name=class_name,
                        vector=vec
                    )
                except Exception as e:
                    print(f"Error inserting row {idx} (class={class_name}): {e}")

    def query(self, text, top_k=5, metric="cosine", where_filter=None, class_name="Document"):
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector}

        query_obj = (
            self.client.query
            .get(class_name, ["*"]) 
            .with_near_vector(near_vector)
            .with_additional(["distance"])
            .with_limit(top_k)
        )
        if where_filter:
            query_obj = query_obj.with_where(where_filter)

        result = query_obj.do()
        return result.get("data", {}).get("Get", {}).get(class_name, [])

    def hybrid_search(self, text, where_filter=None, top_k=5, metric="cosine", class_name="Document"):
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector, "certainty": 0.7} if metric == "cosine" else {"vector": vector}

        query = (
            self.client.query
            .get(class_name, ["*"])
            .with_near_vector(near_vector)
            .with_limit(top_k)
        )
        if where_filter:
            query = query.with_where(where_filter)
        if metric in ["cosine", "dot", "euclidean"]:
            query._additional["distance"] = metric

        result = query.do()
        return result.get("data", {}).get("Get", {}).get(class_name, [])

    def filter_search(self, where_filter, top_k=5, class_name="Document"):
        query = (
            self.client.query
            .get(class_name, ["*"])
            .with_where(where_filter)
            .with_limit(top_k)
        )
        result = query.do()
        return result.get("data", {}).get("Get", {}).get(class_name, [])
