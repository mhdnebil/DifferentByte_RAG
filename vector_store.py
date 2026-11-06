import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from text_chunker import split_text
from pdf_reader import extract_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read PDF
text = extract_text("DifferentByte Company Information.pdf")


# Split into chunks
chunks = split_text(text)


# Generate embeddings
embeddings = model.encode(chunks)

# Convert embeddings to float32
embeddings = np.array(embeddings).astype("float32")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# Store embeddings
index.add(embeddings)


# Save FAISS index
faiss.write_index(index, "document.index")


print("FAISS index created successfully!")

print("Total vectors stored:", index.ntotal)

print("Vector dimensions:", dimension)