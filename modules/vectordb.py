from langchain.vectorstores import Clarifai as Clarifai_vectordb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma, FAISS
from modules.models import *

def create_db(pdfs_folde_path):

    loader = PyPDFDirectoryLoader(pdfs_folde_path)
    pages = loader.load()

    if not pages:
        raise Exception("No pages loaded from PDF documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    docs = text_splitter.split_documents(pages)

    if not docs:
        raise Exception("No Documents created after splitting")
    
    # Extract texts from documents
    texts = [d.page_content for d in docs]

    print(f"Number of texts to embed: {len(texts)}")
    print(f"First text snippet: {texts[0][:100]}")  # Print first 100 characters of the first text

    '''
    # persist_directory = 'db'

    # vector_db = Chroma.from_documents(documents=docs,
    #                                 embedding=clarifai_embedding_model,
    #                                 persist_directory=persist_directory)


    # vector_db = Clarifai_vectordb.from_documents(
    #     user_id=st.secrets["USER_ID"],
    #     app_id=st.secrets["APP_ID"],
    #     documents=docs,
    #     pat=st.secrets["CLARIFAI_PAT"],
    #     number_of_docs=2,
    # )
    if len(docs)<1:
        raise Exception("No Documents created")
    else:
        texts = [d.page_content for d in docs]
        embeddings = clarifai_embedding_model.embed_documents(texts)
        if not embeddings or len(embeddings) == 0:
            raise Exception("Embedding failed, no embeddings generated")
        vector_db = FAISS.from_documents(docs, clarifai_embedding_model)
    return vector_db
    '''
    try:
        # Embed the texts
        embeddings = clarifai_embedding_model.embed_documents(texts)
    except Exception as e:
        raise Exception(f"Error during embedding: {str(e)}")

    if not embeddings:
        raise Exception("Embedding failed, no embeddings generated")
    if len(embeddings) != len(texts):
        raise Exception(f"Number of embeddings ({len(embeddings)}) does not match number of texts ({len(texts)})")

    # Create vector store from documents and embeddings
    try:
        vector_db = FAISS.from_documents(docs, clarifai_embedding_model)
    except Exception as e:
        raise Exception(f"Error creating vector DB: {str(e)}")
    
    return vector_db
