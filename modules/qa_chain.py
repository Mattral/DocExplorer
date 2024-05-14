from langchain.chains import RetrievalQA
from langchain.retrievers import QdrantSparseVectorRetriever

def create_chain(local_llm, vectordb):
    retriever = QdrantSparseVectorRetriever(
        client=vectordb,
        collection_name="sparse_collection",
        vector_name="sparse_vector",
        embedding_function=encoder.encode,
        distance_func='cosine'
    )

    qa = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    qa.combine_documents_chain.llm_chain.prompt.template = '''
    Your name is PTT.
    Use the following pieces of context to answer the user's question and that
    information consist of provided pdf details by user.
    If you don't know the answer, just apologize and say that you couldn't
    find relevant content to answer this question, don't try to
    make anything from yourself just use the provided context. And always answer
    in a conversational friendly way. And don't make any Follow-up Question and don't
    provide any further information just answer the question.
    Always answer from the perspective of being Ash.
    ----------------
    {context}

    Question: {question}
    Helpful Answer:'''
    return qa
