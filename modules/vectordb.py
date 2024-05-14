from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import streamlit as st

# Load the embedding model
encoder = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
print("Embedding model loaded...")

def create_db(pdfs_folder_path):
    # Load PDF documents
    loader = PyPDFDirectoryLoader(pdfs_folder_path)
    pages = loader.load()
    
    if not pages:
        raise Exception("No pages loaded from PDF documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    if not docs:
        raise Exception("No Documents created after splitting")

    print(f"Number of documents created: {len(docs)}")
    print(f"First document: {docs[0].page_content[:100]}")  # Print first 100 characters of the first doc

    # Extract texts from documents
    texts = [d.page_content for d in docs]

    print(f"Number of texts to embed: {len(texts)}")
    print(f"First text snippet: {texts[0][:100]}")  # Print first 100 characters of the first text

    try:
        # Embed the texts using SentenceTransformer
        embeddings = encoder.encode(texts)
    except Exception as e:
        raise Exception(f"Error during embedding: {str(e)}")

    if embeddings is None or len(embeddings) == 0:
        raise Exception("Embedding failed, no embeddings generated")
    if len(embeddings) != len(texts):
        raise Exception(f"Number of embeddings ({len(embeddings)}) does not match number of texts ({len(texts)})")

    # Create Qdrant client and collection
    client = QdrantClient(path="./qdrant_db")
    client.recreate_collection(
        collection_name="my_facts",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

    # Prepare documents for upload
    doc_metadata = [
        models.Record(
            id=i,
            vector=embedding.tolist(),
            payload={"text": text}
        )
        for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    # Upload documents to Qdrant
    try:
        client.upload_records(collection_name="my_facts", records=doc_metadata)
    except Exception as e:
        raise Exception(f"Error uploading records to Qdrant: {str(e)}")

    print("Records uploaded successfully to Qdrant")

    return client

# Ensure to use st.rerun instead of st.experimental_rerun
def read_docs():
    with st.spinner("Reading Documents........"):
        if not (st.session_state.get("chain_created")) and st.session_state.get("processed"):
            db = create_db("uploads")
            st.session_state["qa"] = create_chain(llm, db)  # Replace llm with your actual LLM
            st.session_state["chain_created"] = True
            st.rerun()
