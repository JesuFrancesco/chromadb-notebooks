import os
import argparse
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a PDF document into ChromaDB")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document")
    parser.add_argument(
        "collection_name", type=str, help="Name of the ChromaDB collection"
    )
    args = parser.parse_args()

    # Crear embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Configurar la conexión a ChromaDB
    conn = chromadb.HttpClient(
        host=os.getenv("CHROMADB_HOST", "localhost"),
        port=int(os.getenv("CHROMADB_PORT", "8000")),
    )
    print(conn.heartbeat())

    chroma_vs = Chroma(
        collection_name=args.collection_name,
        embedding_function=embeddings,
        client=conn,
    )

    # Cargar el documento PDF
    pdf_path = args.pdf_path
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Dividir el documento en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(docs)

    # Agregar documentos a la colección
    print(f"Cargando documentos en la colección '{args.collection_name}'...")
    chroma_vs.add_documents(docs)

    print(f"Documentos cargados en la colección '{args.collection_name}' de ChromaDB.")
