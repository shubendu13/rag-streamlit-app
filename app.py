import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
import torch

# Set model device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model and vector DB
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Define per-product summarization prompt
summary_prompt = PromptTemplate.from_template("""
You are a helpful product assistant.

Summarize the following product in 40-50 words. Use bullet points for key features. 
Do not describe obvious visual details since the user will see the image.

### Product:
{context}

### Summary:
""")

# Load LLM (TinyLlama)
llm = HuggingFacePipeline.from_model_id(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1
)

# Build the summarization chain
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

st.title("🧠 Product Search (Text + Image RAG)")
query = st.text_input("Enter your product-related query:")

# Convert s3 path to HTTP
def s3_to_http_url(s3_path):
    if s3_path.startswith("s3://"):
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return s3_path

# Main logic
if query:
    # Use retriever directly
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    st.subheader("🔎 Retrieved Product Summaries")

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        item_id = meta.get("item_id", "Unknown")
        image_url = s3_to_http_url(meta.get("image_path", ""))

        # Run summary chain on each document
        summary = summary_chain.run(context=doc.page_content)

        # Display results
        st.markdown(f"### 🛒 Product {i}: {item_id}")
        st.markdown("*🤖 Summary:*")
        st.write(summary)

        st.markdown("*📄 Description:*")
        st.write(doc.page_content)

        if image_url:
            st.image(image_url, caption=f"Item ID: {item_id}", use_column_width=True)
        else:
            st.warning("🚫 Image not available.")

        st.markdown("---")