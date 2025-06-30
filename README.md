# 🧠 RAG-Based Product Search (Text + Image) with Streamlit + LangChain + ChromaDB

This project implements a **Retrieval-Augmented Generation (RAG)** system that:
- Captions product images using **BLIP-2**
- Embeds product data using **E5-small-v2**
- Stores everything in **ChromaDB**
- Answers product-related queries using **TinyLlama**
- Displays results with product image, text, and a natural language response via **Streamlit**

---

## 📌 Features

| Feature                    | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 🖼️ Image Captioning       | Uses `BLIP-2` to auto-generate captions for images                          |
| 🔤 Text Embedding         | Uses `E5-small-v2` to embed product names, bullet points, descriptions     |
| 🧲 Vector Store           | Stores embeddings in **ChromaDB** with image paths                          |
| 💬 Open-source LLM        | Uses **TinyLlama** for query generation and response (via LangChain)        |
| 🧑‍💻 Streamlit UI         | Simple interface for querying and visualizing results                        |
| 🐳 Dockerized             | Easily deploy the app in a Docker container                                 |
| 📦 Modular Codebase       | Separate scripts for preprocessing, indexing, and UI                        |

---

## 📁 Project Structure

```
project-root/
├── app.py # Streamlit UI with RAG pipeline
├── prepare_index.py # Embeds text + images and stores in Chroma
├── data_preprocess.py # Extracts and cleans JSON data
├── requirements.txt
├── Dockerfile
├── README.md
├── .github/  Future enhancements for ci/cd deployments
│ └── workflows/
│ └── deploy.yml # GitHub Actions for CI/CD
```


---

## 🚀 Quickstart (Local)

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/yourname/rag-product-search.git
cd rag-product-search
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Data
```bash
python prepare_index.py
# Or run via Streamlit UI as Rebuild Index
```

### 4️⃣ Launch the App
```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### 1️⃣ Build Docker Image
```
docker build -t rag-app .
```

### 2️⃣ Run the App
```
docker run -p 8501:8501 rag-app
```

---

## Query Example
```
#### Query:
“white Nike t-shirt for a child”

#### Output:

🔤 Matched via text or image caption

🖼 Image displayed

🤖 TinyLlama generates a final answer
```
---

## Future Enhancements - GitHub Actions

```
To enable CI/CD with Docker:

Add .github/workflows/deploy.yml

Use GitHub secrets to store DockerHub or AWS credentials

Auto-build Docker image on main push or schedule```