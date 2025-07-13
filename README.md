# 🧠 RAG Product Search App (Text + Image)

This project is a full-stack **Retrieval-Augmented Generation (RAG)** application to search product data using **text queries** and **image captions**. It's powered by:

- 🔍 BLIP-2 for image captioning
- ✍️ `intfloat/e5-base-v2` for text + caption embeddings
- 🧠 ChromaDB as the vector store
- 🖼️ Streamlit as the web interface
- 🐳 Docker & Docker Compose for containerized deployment

---

## 📌 Features

| Feature                                 | Description                                                              |
| --------------------------------------- | ------------------------------------------------------------------------ |
| 🧠 Retrieval-Augmented Generation (RAG) | Combines dense vector search with generative capabilities                |
| 🔤 Natural Language Search              | Search using plain English queries like "red Nike hoodie for women"      |
| 🖼️ Image + Text Matching               | Matches both product descriptions and image captions                     |
| 🪄 BLIP-2 Captioning                    | Uses BLIP-2 to auto-generate captions from product images                |
| 🔎 E5 Embeddings                        | Uses `intfloat/e5-base-v2` to embed both text and image captions         |
| 📦 ChromaDB Vector Store                | Efficient local similarity search with ChromaDB                          |
| 🔁 Deduplication Logic                  | Avoids duplicate results when both text and image match same item        |
| 💡 Result Merging                       | Merges matched text and caption content for final display and generation |
| ⚙️ Modular Pipeline                     | Separate scripts for indexing and app UI — easy to debug and scale       |
| 📺 Streamlit UI                         | Clean interface for querying and displaying results                      |
| 🐳 Dockerized                           | Reproducible, portable setup using Docker and Docker Compose             |
| ☁️ EC2-Ready                            | Works out of the box on AWS EC2 with volume mounting and exposed ports   |

---

## 📁 Project Structure

```
├── app.py # Streamlit app UI
├── prepare_index.py # Generates BLIP-2 captions and embeddings
├── requirements.txt # Python packages
├── Dockerfile # Environment + install steps
├── docker-compose.yml # Runtime definition for app
├── .dockerignore # Ignore large/unneeded files from image
├── abo-images-small/ # Your image folder (mounted, not copied)
├── image_metadata.csv # Maps image_id → relative path
└── README.md # This file
```


---



---

## 🐳 Docker Setup Instructions (EC2 or local)

### ✅ 1. SSH into EC2 and clone the project

```bash
ssh -i your-key.pem ubuntu@<your-ec2-ip>
git clone https://github.com/your-username/your-repo.git
cd your-repo/
```

### ✅ 2. Build Docker image
```bash
docker-compose build
```

### ✅ 3. Run indexing once
```bash
docker-compose run --rm rag python prepare_index.py
```

### ✅ 4. Launch the Streamlit app
```bash
docker-compose up -d
```

### ✅ 5. Visit the URL
```
http://<your-ec2-public-ip>:8501
```

---

## 🧹 Debugging

Check logs:

```bash
docker-compose logs -f
```


Rebuild if needed:

```bash
docker-compose down
docker-compose up --build -d
```

---

## ✅ Done!

You're now running a fully functional RAG + Streamlit app with image captioning on EC2 using Docker.
