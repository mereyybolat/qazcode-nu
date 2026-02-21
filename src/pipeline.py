from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalPipeline:
    protocols: list[dict[str, Any]]
    embeddings: np.ndarray
    encoder: SentenceTransformer

    @classmethod
    def from_artifact(cls, artifact_path: str | Path) -> "RetrievalPipeline":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        with path.open("rb") as f:
            artifact = pickle.load(f)

        protocols = artifact["protocols"]
        embeddings = artifact["embeddings"]
        model_dir = artifact["encoder"]["model_dir"]

        encoder = SentenceTransformer(model_dir, local_files_only=True)
        return cls(protocols=protocols, embeddings=embeddings, encoder=encoder)

    def retrieve(self, symptoms: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vec = self.encoder.encode([symptoms], convert_to_numpy=True).astype("float32")
        query_vec = query_vec[0]

        emb = self.embeddings
        query_norm = np.linalg.norm(query_vec) + 1e-12
        emb_norm = np.linalg.norm(emb, axis=1) + 1e-12
        scores = np.dot(emb, query_vec) / (emb_norm * query_norm)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in top_idx:
            results.append(
                {
                    "score": float(scores[idx]),
                    "protocol_id": self.protocols[idx].get("protocol_id"),
                    "title": self.protocols[idx].get("title", ""),
                    "icd_codes": self.protocols[idx].get("icd_codes", []),
                    "text": self.protocols[idx].get("text", ""),
                }
            )
        return results
