import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch

# Set model device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
BUCKET_NAME = "shubendu-rag-llm-app-bucket"

# Load embedding model and vector DB
#Large model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
#smaller model
#embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load open-source LLM (TinyLlama or similar)
llm = HuggingFacePipeline.from_model_id(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Smaller model
    #"mistralai/Mistral-7B-Instruct-v0.1", #Large model
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

st.title("🧠 Product Search (Text + Image RAG)")

query = st.text_input("Enter your product-related query:")

# Convert s3://... path to https://... path
def s3_to_http_url(s3_path):
    if s3_path.startswith("s3://"):
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return s3_path

if query:
    full_query = f"Answer as a product specialist. Write concise answer in a formatted manner and use" \
                 f" bullet points whenever necessary to explain the product item. Also - {query}"
    result = qa(full_query)
    st.subheader("🤖 Answer")
    st.write(result["result"])

    st.subheader("🔎 Retrieved Matches")
    for doc in result["source_documents"]:
        meta = doc.metadata
        st.markdown(f"**Item ID:** `{meta.get('item_id', 'Unknown')}`")

        if meta.get("image_path"):
            image_url = s3_to_http_url(meta["image_path"])
            st.image(image_url, caption=f"{meta['item_id']}")
        else:
            st.warning("Image not available.")

        st.markdown("**Description:**")
        st.write(doc.page_content)
        st.markdown("---")
