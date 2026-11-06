import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from text_chunker import split_text
from pdf_reader import extract_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load original document
text = extract_text("DifferentByte Company Information.pdf")

# Generate chunks
chunks = split_text(text)


# Load FAISS index
index = faiss.read_index("document.index")


def search_documents(query, top_k=2):

    # Convert question into embedding
    query_embedding = model.encode([query])

    # Convert to float32
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    print("\nRelevant Documents:\n")

    for i, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        print(f"Result {i + 1}")
        print("Distance:", distances[0][i])
        print("Content:", chunks[idx])
        print("-" * 50)


if __name__ == "__main__":

    question = input("Ask a question about your PDF: ")

    search_documents(question)