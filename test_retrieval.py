from src.retriever import Retriever

retriever = Retriever()

query = "pregnant woman high blood pressure low platelets"

results = retriever.search(query, top_k=3)

for r in results:
    print("ICD:", r["icd_codes"])
    print("Title:", r["title"])
    print()