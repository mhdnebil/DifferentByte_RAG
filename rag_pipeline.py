import os
import faiss
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from pdf_reader import extract_text
from text_chunker import split_text


# Load environment variables
load_dotenv()

# Initialize Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract document text
text = extract_text("DifferentByte Company Information.pdf")

# Split into chunks
chunks = split_text(text)

# Load FAISS index
index = faiss.read_index("document.index")


def retrieve_context(question, top_k=2):

    query_embedding = model.encode([question])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx != -1:
            retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)


def generate_answer(question):

    # Retrieve relevant information
    context = retrieve_context(question)

    # Create prompt
    prompt = f"""
    Answer the question using only the provided context.

    If the answer is not available in the context,
    say that the information is not available.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Send request to Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type exit to stop): ")

        if question.lower() == "exit":
            break

        answer = generate_answer(question)

        print("\nAI Answer:")
        print(answer)