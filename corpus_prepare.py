import json
import pickle
from pathlib import Path


INPUT_PATH = Path("data/corpus.json")
OUTPUT_PATH = Path("data/processed_corpus.pkl")


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
        else:
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    return data


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text


def main():
    print("Loading corpus...")
    protocols = load_corpus(INPUT_PATH)

    print("Total protocols loaded:", len(protocols))

    processed = []

    for p in protocols:
        cleaned_text = clean_text(p["text"])

        processed.append({
            "protocol_id": p["protocol_id"],
            "source_file": p.get("source_file", ""),
            "title": p.get("title", ""),
            "icd_codes": p.get("icd_codes", []),
            "text": cleaned_text
        })

    print("Saving processed corpus...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(processed, f)

    print("Done.")


if __name__ == "__main__":
    main()