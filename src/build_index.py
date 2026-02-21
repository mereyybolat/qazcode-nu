import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sentence-transformer embeddings and save a deployable model.pkl artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed_corpus.pkl"),
        help="Path to processed corpus pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model.pkl"),
        help="Path to output model artifact pickle.",
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=Path("data/embeddings.npy"),
        help="Optional path to save raw embeddings (.npy).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/paraphrase-multilingual-MiniLM-L12-v2"),
        help="Local path to sentence-transformer model directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.",
    )
    return parser.parse_args()


def load_processed_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Processed corpus must be a non-empty list of protocol records.")
    return data


def clean_protocol_record(record: dict) -> dict:
    return {
        "protocol_id": record.get("protocol_id"),
        "source_file": record.get("source_file", ""),
        "title": record.get("title", ""),
        "icd_codes": record.get("icd_codes", []),
        "text": record.get("text", ""),
    }


def main() -> None:
    args = parse_args()

    print(f"Loading processed corpus: {args.input}")
    protocols = [clean_protocol_record(p) for p in load_processed_corpus(args.input)]
    texts = [p["text"] for p in protocols]
    print(f"Protocols loaded: {len(protocols)}")

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Local sentence-transformer model not found: {args.model_dir}. "
            "Download/package it before running this step."
        )

    print(f"Loading local embedding model: {args.model_dir}")
    model = SentenceTransformer(str(args.model_dir), local_files_only=True)

    print("Encoding protocol texts...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_output, embeddings)
    print(f"Saved embeddings array: {args.embeddings_output} shape={embeddings.shape}")

    artifact = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": {
            "type": "sentence-transformers",
            "model_dir": str(args.model_dir),
            "local_files_only": True,
            "embedding_dim": int(embeddings.shape[1]),
        },
        "protocol_count": len(protocols),
        "protocols": protocols,
        "embeddings": embeddings,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact: {args.output}")


if __name__ == "__main__":
    main()
