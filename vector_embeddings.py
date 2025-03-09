# -*- coding: utf-8 -*-
"""vector_embeddings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LzYh0ETxlGzlMaU8Mv5CTnYVNMwEbHxo
"""

!pip install chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer
import json

import json
with open("/content/drive/MyDrive/document_chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)  # Now all_chunks contains list of dictionaries

all_chunks[0]

from chromadb.utils import embedding_functions

# Initialize embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize ChromaDB
chroma_client = chromadb.Client()
# chroma_client = chroma_client.PersistentClient(path="/content/drive/MyDrive/chroma_data")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create a collection
collection = chroma_client.create_collection(
    name="ai_research",
    embedding_function=sentence_transformer_ef
)



# Add chunks to the collection
ids = [f"chunk_{i}" for i in range(len(all_chunks))]
texts = [chunk["text"] for chunk in all_chunks]
metadatas = [{"source": chunk["source"], "title": chunk["title"]} for chunk in all_chunks]

# Add in batches to avoid memory issues
batch_size = 100
for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    collection.add(
        ids=ids[i:end_idx],
        documents=texts[i:end_idx],
        metadatas=metadatas[i:end_idx]
    )

print(f"Added {len(ids)} chunks to vector database")

# Export the database for later use
# chroma_client.persist()

query_text = "How does DeepSeek-V3 handle MoE layers?"

# Search for relevant chunks
results = collection.query(
    query_texts=[query_text],  # Convert query to embedding & search
    n_results=3  # Retrieve top 3 most relevant chunks
)

# Print results
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n🔹 **Result {i+1}**")
    print(f"📄 **Source**: {meta['source']}")
    print(f"🏷 **Title**: {meta['title']}")
    print(f"📜 **Text**: {doc}")



