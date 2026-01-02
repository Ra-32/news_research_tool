from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
import langchain
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import pickle
import time
load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
file_path = "faiss_store_hg.pkl"
parser=StrOutputParser()
if "processed" not in st.session_state:
    st.session_state.processed = False

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-genration",
    huggingfacehub_api_token=hf_token,
)
model=ChatHuggingFace(llm=llm)
st.title("News Research Tool 🤖 🎬")

st.sidebar.title("News article URLs 🎞️")
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url=st.sidebar.button("process urls")

main_placeholder=st.empty()

if process_url:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started...✅✅")
    doc=loader.load()
    # text spliter
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    main_placeholder.text("text splitting started....✅✅")
    chunks=splitter.split_documents(doc)
    #vector store
    llm_embedding=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
    )
    vector_store=FAISS.from_documents(chunks,llm_embedding)
    main_placeholder.text("embedding stated ...✅✅")
    time.sleep(2)

    vector_store.save_local("faiss_index")
    query=main_placeholder.text_input("Question:")
    vector_store.save_local("faiss_index")
    st.session_state.processed = True
    st.success("URLs processed successfully!")

# query = main_placeholder.text_input("Question:",key="question_input")

if st.session_state.processed:
    query = main_placeholder.text_input("Question:",key="question_input")

    if st.button("Submit") and query:
        llm_embedding = HuggingFaceEndpointEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )

        vectorstore = FAISS.load_local(
            "faiss_index",
            llm_embedding,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        retrieved_docs = retriever.invoke(query)

       
        page_contents = []
        source_links = []

        for doc in retrieved_docs:
            
            source_links.append(doc.metadata.get("source"))

        
        
        def get_context(doc):
            context="\n\n".join(doc.page_content for doc in retrieved_docs)
            return context

        
        
        template=PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided  context.
            If the context is insufficient, just say you don't know.
            

            {context}
            
            Question: {question}
             """,
            input_variables=['context','question'],
            )
        
        # prompt=template.invoke({'context':get_context(retrieved_docs),'question':query})
        # answer=model.invoke(prompt)
        # if answer:
        #     st.write(answer.content)
        #     st.write(source_links[0])

        parallel_chain=RunnableParallel({
            'context':retriever | RunnableLambda(get_context),
            'question':RunnablePassthrough()
        })

        main_chain=parallel_chain | template | model |parser

        answer=main_chain.invoke(query)
        if answer:
            st.write(answer)
            st.write(source_links[0])
