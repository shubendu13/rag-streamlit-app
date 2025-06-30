import streamlit as st
#from langchain.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from PIL import Image
import os
import torch
from data_preprocess import prepare_index

# Set model device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embeddings and vector DB from Google Drive
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
vectordb = Chroma(
    persist_directory="/content/drive/My Drive/Colab Notebooks/ShopTalkData/chroma_db",
    embedding_function=embeddings
)

# Load the open-source LLM (TinyLlama or any other)
llm = HuggingFacePipeline.from_model_id(
    #"TinyLlama/TinyLlama-1.1B-Chat-v1.0", #Small model
    "mistralai/Mistral-7B-Instruct-v0.1", #Large model
    task="text-generation",
    device=0 if device == "cuda" else -1
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

with st.sidebar.expander("⚙️ Settings", expanded=False):
    st.markdown("Manually rebuild your ChromaDB index if data has changed.")

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding ChromaDB index..."):
            prepare_index()
        st.success("✅ Index rebuilt successfully!")

st.title("🧠 Product Search (Text + Image RAG)")

query = st.text_input("Enter your product-related query:")

if query:
    result = qa(query)
    st.subheader("🤖 Answer")
    st.write(result["result"])

    st.subheader("🔎 Retrieved Matches")
    for doc in result["source_documents"]:
        meta = doc.metadata
        st.markdown(f"**Item ID:** `{meta['item_id']}`")
        if meta.get("image_path") and os.path.exists(meta["image_path"]):
            st.image(meta["image_path"], caption=f"{meta['item_id']}")
        else:
            st.warning("Image not found.")
        st.write(doc.page_content)
