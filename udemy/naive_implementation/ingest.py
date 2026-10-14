import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv
load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / os.getenv("DB_NAME", "vector_db"))
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / os.getenv("KNOWLEDGE_BASE", "knowledge-base"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("API_KEY", "ollama")
    

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,    base_url=BASE_URL, 
    api_key=API_KEY,
    check_embedding_ctx_length=False )


def fetch_documents():
    print("Fetching documents from knowledge base")
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    print("Creating chunks from documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    print("Creating embeddings for chunks")
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    print("Starting ingestion process")
    documents = fetch_documents()
    print(f"Fetched {len(documents):,} documents from knowledge base")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks):,} chunks from documents")
    create_embeddings(chunks)
    print("Ingestion complete")
