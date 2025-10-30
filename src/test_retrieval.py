from embeddings import model  
from retrieval import semantic_search

# Example query
query = "What is the procedure for filing a lawsuit?"
domain = "law"

# Call the semantic search
results = semantic_search(query=query, domain=domain, top_k=5, embed_model=model)

# Print results
print("Top relevant chunks:")
for i, chunk in enumerate(results, 1):
    print(f"{i}. {chunk}\n")
