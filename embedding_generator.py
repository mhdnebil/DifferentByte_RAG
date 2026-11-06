from sentence_transformers import SentenceTransformer
from text_chunker import split_text
from pdf_reader import extract_text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract PDF text
text = extract_text("DifferentByte Company Information.pdf")

# Split text into chunks
chunks = split_text(text)

# Generate embeddings
embeddings = model.encode(chunks)

print("Total Chunks:", len(chunks))
print("Embedding Shape:", embeddings.shape)

for index, embedding in enumerate(embeddings):

    print(f"\nChunk {index + 1}")
    print("Vector Dimensions:", len(embedding))
    print("First 5 Values:", embedding[:5])