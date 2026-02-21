import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    print("Loading processed corpus...")
    with open("data/processed_corpus.pkl", "rb") as f:
        protocols = pickle.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [p["text"] for p in protocols]

    print("Encoding texts...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("Saving embeddings...")
    np.save("data/embeddings.npy", embeddings)

    print("Saving protocols metadata...")
    with open("data/protocols.pkl", "wb") as f:
        pickle.dump(protocols, f)

    print("Done.")


if __name__ == "__main__":
    main()