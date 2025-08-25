import pandas as pd
import weaviate
import json

class DMSIndexer:
    def __init__(self, weaviate_client: weaviate.Client, embedding_model):
        self.client = weaviate_client
        self.model = embedding_model
        self.flattened_df = pd.DataFrame()

    def import_from_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and len(data) == 1:
                self.dms_tree = list(data.values())[0]
            elif isinstance(data, list):
                self.dms_tree = data
            else:
                raise ValueError("Unexpected JSON structure: top-level must be a list or dict with single key")
        return self.dms_tree

    def import_from_db(self, conn):
        repos = pd.read_sql("SELECT * FROM repos", conn).to_dict(orient="records")
        folders = pd.read_sql("SELECT * FROM folders", conn).to_dict(orient="records")
        documents = pd.read_sql("SELECT * FROM documents", conn).to_dict(orient="records")
        files = pd.read_sql("SELECT * FROM files", conn).to_dict(orient="records")
        for repo in repos:
            repo["folders"] = [f for f in folders if f["repo_id"] == repo["repo_id"]]
            for folder in repo["folders"]:
                folder["documents"] = [d for d in documents if d["folder_id"] == folder["folder_id"]]
                for doc in folder["documents"]:
                    doc["files"] = [f for f in files if f["document_id"] == doc["document_id"]]
        self.dms_tree = repos
        return self.dms_tree

    def import_from_fs(self, root_path):
        import os
        dms_tree = []
        for repo_name in os.listdir(root_path):
            repo_path = os.path.join(root_path, repo_name)
            if not os.path.isdir(repo_path):
                continue
            repo = {"repo_id": repo_name, "repo_name": repo_name, "folders": []}
            for folder_name in os.listdir(repo_path):
                folder_path = os.path.join(repo_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                folder = {"folder_id": folder_name, "folder_name": folder_name, "documents": []}
                for doc_name in os.listdir(folder_path):
                    doc_path = os.path.join(folder_path, doc_name)
                    if not os.path.isdir(doc_path):
                        continue
                    doc = {"document_id": doc_name, "title": doc_name, "files": []}
                    for fname in os.listdir(doc_path):
                        fpath = os.path.join(doc_path, fname)
                        if os.path.isfile(fpath):
                            with open(fpath, "r", encoding="utf-8") as f:
                                doc["files"].append({"file_name": fname, "content": f.read()})
                    folder["documents"].append(doc)
                repo["folders"].append(folder)
            dms_tree.append(repo)
        self.dms_tree = dms_tree
        return dms_tree

    def flatten_dms(self):
        rows = []
        def flatten_folder(folder, repo_id, repo_name):
            for doc in folder.get("documents", []):
                row = {
                    "document_id": doc.get("document_id"),
                    "repo_id": repo_id,
                    "repo_name": repo_name,
                    "folder_id": folder.get("folder_id"),
                    "folder_name": folder.get("folder_name"),
                    "title": doc.get("title"),
                    "author": doc.get("author"),
                    "tags": doc.get("tags"),
                    "category": doc.get("category"),
                    "content": doc.get("content")
                }
                files = doc.get("files", [])
                row["files_content"] = " | ".join(f"{f.get('file_name')}: {f.get('content','')}" for f in files) if files else None
                rows.append(row)
            for sub in folder.get("folders", []):
                rows.extend(flatten_folder(sub, repo_id, repo_name))
            return rows
        if isinstance(self.dms_tree, dict) and "repositories" in self.dms_tree:
            repos = self.dms_tree["repositories"]
        elif isinstance(self.dms_tree, list):
            repos = self.dms_tree
        else:
            raise ValueError("Unexpected DMS tree structure")
        for repo in repos:
            repo_id = repo.get("repo_id")
            repo_name = repo.get("repo_name")
            for folder in repo.get("folders", []):
                rows.extend(flatten_folder(folder, repo_id, repo_name))
        self.flattened_df = pd.DataFrame(rows)
        return self.flattened_df

    def compute_embeddings(self):
        def flatten_row(row):
            parts = []
            for key, value in row.items():
                if key == "embedding" or value is None:
                    continue
                if isinstance(value, list):
                    val_str = ", ".join(map(str, value))
                else:
                    val_str = str(value)
                parts.append(f"{key}: {val_str}")
            return " | ".join(parts)

        embeddings = []
        for idx, row in self.flattened_df.iterrows():
            text_repr = flatten_row(row)
            emb = self.model.encode(text_repr, convert_to_numpy=True)
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            elif not isinstance(emb, list):
                emb = list(emb)
            embeddings.append([float(x) for x in emb])
            print(f"Computed embedding for row {idx}, document_id={row.get('document_id')}, length={len(embeddings[-1])}")
        self.flattened_df["embedding"] = embeddings
        return self.flattened_df


    def reset_schema(self):
        classes = [c["class"] for c in self.client.schema.get()["classes"]]
        if "Document" in classes:
            self.client.schema.delete_class("Document")
        class_obj = {
            "class": "Document",
            "vectorizer": "none",
            "properties": [
                {"name": "document_id", "dataType": ["text"]},
                {"name": "repo_id", "dataType": ["text"]},
                {"name": "repo_name", "dataType": ["text"]},
                {"name": "folder_id", "dataType": ["text"]},
                {"name": "folder_name", "dataType": ["text"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "author", "dataType": ["text"]},
                {"name": "tags", "dataType": ["text[]"]},
                {"name": "category", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "files_content", "dataType": ["text"]},
                {"name": "embedding", "dataType": ["blob"]}
            ]
        }
        self.client.schema.create_class(class_obj)

    def insert_documents(self, batch_size=50):
        self.reset_schema()
        with self.client.batch(batch_size=batch_size) as batch:
            for idx, row in self.flattened_df.iterrows():
                data_obj = row.to_dict()
                vec = data_obj.pop("embedding", None)
                if vec is None or len(vec) == 0:
                    print(f"Skipping document {row.get('document_id')} due to missing embedding")
                    continue
                vec = [float(x) for x in vec]
                try:
                    batch.add_data_object(data_object=data_obj, class_name="Document", vector=vec)
                    print(f"Queued document {idx}, document_id={row.get('document_id')} for insertion, Vector {vec}")
                except Exception as e:
                    print(f"Error inserting document {idx}, document_id={row.get('document_id')}: {e}")
            batch.flush()
        print(f"{len(self.flattened_df)} documents processed for insertion.")


    def search(self, query, top_k=5, where_filter=None, certainty=0.2):
        vector = self.model.encode(query, convert_to_numpy=True).tolist()
        near_vector = {"vector": vector, "certainty": certainty}
        print(f"Searching with vector of length {len(vector)} for query: {query}")
        query_obj = self.client.query.get("Document", ["*"]).with_near_vector(near_vector).with_limit(top_k)
        if where_filter:
            query_obj = query_obj.with_where(where_filter)
        result = query_obj.do()
        hits = result.get("data", {}).get("Get", {}).get("Document", [])
        print(f"Search returned {len(hits)} hits")
        return [{"document_id": h.get("document_id"), "distance": h.get("_additional", {}).get("distance")} for h in hits]

