"""
model.py: Defines Hugging Face CausalLM setup and ChromaDB RAG Vector Store.
"""
import os
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

MODEL_NAME = "gpt2"  # Lightweight baseline for CPU/demo fine-tuning
CHROMA_DIR = "chroma_db"

def get_fine_tuning_model():
    """Initializes Hugging Face model and tokenizer for training/inference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def initialize_vector_db(openai_api_key: str = None) -> Chroma:
    """Sets up ChromaDB vector database with ESL pedagogical rules."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY", "mock_key"))
    
    curriculum_docs = [
        Document(
            page_content="A1/A2 Scaffolding: Keep sentences under 10 words. Correct 1 error per turn gently using re-casting.",
            metadata={"level": "Beginner"}
        ),
        Document(
            page_content="B1/B2 Scaffolding: Use complex tenses (present perfect, conditionals). Focus on fluency and vocabulary expansion.",
            metadata={"level": "Intermediate"}
        )
    ]
    
    vector_db = Chroma.from_documents(
        documents=curriculum_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_db