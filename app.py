import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Load Environment Variables
load_dotenv()

# 2. Page Configuration
st.set_page_config(page_title="Dokitor", page_icon="")
st.title("Dokitor - Your Medical Assistant Chatbot")
st.caption("Knowledge Base: Where There Is No Doctor (Handbook)")

# 3. Load Knowledge Base (Cached for Speed)
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./medical_db", embedding_function=embeddings)
    return db

vector_db = load_vector_db()

# 4. Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# 5. Create prompt template
template = """Use the following pieces of context to answer the question. If the answer is not in the context, say you don't know.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

# 7. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the handbook..."):
            response = qa_chain.invoke(user_query)
            answer = response["result"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})