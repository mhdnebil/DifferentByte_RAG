import os
import streamlit as st
import faiss
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Configuration
load_dotenv()

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions about its content.")


# Initialize clients
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found in .env")
    st.stop()

client = Groq(api_key=api_key)


# Extract PDF text
def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Text chunking
def split_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Process uploaded PDF
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file:

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            text = extract_pdf_text(uploaded_file)

            if not text.strip():
                st.error("No readable text found in PDF.")
                st.stop()

            chunks = split_text(text)

            embeddings = model.encode(chunks)

            embeddings = np.array(
                embeddings
            ).astype("float32")

            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.processed = True

        st.success("Document processed successfully!")


# Ask questions
if st.session_state.get("processed", False):

    st.divider()

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        if question.strip():

            with st.spinner("Finding answer..."):

                query_embedding = model.encode(
                    [question]
                )

                query_embedding = np.array(
                    query_embedding
                ).astype("float32")

                distances, indices = st.session_state.index.search(
                    query_embedding,
                    min(3, st.session_state.index.ntotal)
                )

                context = "\n\n".join(
                    st.session_state.chunks[i]
                    for i in indices[0]
                    if i != -1
                )

                prompt = f"""
                Answer using only the provided document context.

                If the answer is unavailable in the context,
                clearly say that the information is not available.

                Context:
                {context}

                Question:
                {question}
                """

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

                answer = response.choices[0].message.content

            st.subheader("AI Answer")
            st.write(answer)

        else:
            st.warning("Please enter a question.")