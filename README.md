# DifferentByte RAG Application

## Project Overview
A Retrieval-Augmented Generation application that allows users
to upload PDF documents and ask questions based on their content.

## Technologies Used
- Python
- Streamlit
- Groq API
- Sentence Transformers
- FAISS
- PyPDF

## Features
- PDF text extraction
- Text chunking
- Embedding generation
- Vector similarity search
- LLM-based answer generation

## Installation

Install dependencies:

pip install -r requirements.txt

## Environment Configuration

Create a .env file and add:

GROQ_API_KEY=your_api_key

## Run Application

streamlit run app.py