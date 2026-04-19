import numpy as np
from FlagEmbedding import BGEM3FlagModel

from .professor_loader import professor_index_text


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"[Embedder] Loading {model_name} via FlagEmbedding...", flush=True)
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.professors: list[dict] = []
        self.matrix: np.ndarray | None = None
        print("[Embedder] Model loaded.", flush=True)

    def build_index(self, professors: list[dict]) -> None:
        self.professors = professors
        texts = [professor_index_text(p) for p in professors]
        print(f"[Embedder] Encoding {len(texts)} profiles...", flush=True)
        output = self.model.encode(texts, batch_size=8, max_length=512)
        vecs = output["dense_vecs"].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs / np.maximum(norms, 1e-9)
        print("[Embedder] Index ready.", flush=True)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        output = self.model.encode([query], max_length=512)
        q_emb = output["dense_vecs"].astype(np.float32).flatten()
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)
        scores = (self.matrix @ q_emb).squeeze()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.professors[i], "similarity_score": float(scores[i])}
            for i in indices
        ]
