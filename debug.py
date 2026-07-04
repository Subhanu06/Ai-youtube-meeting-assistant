from core.vector_store import load_vector_store

vector_store = load_vector_store()

query = "pricing strategy"

docs = vector_store.similarity_search_with_score(query, k=5)

print("\n========== SCORE DEBUG ==========")

for doc, score in docs:
    print("\nSCORE:", score)
    print(doc.page_content[:200])