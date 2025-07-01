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
├── prepare_index.py # Embeds text + images and stores in Chroma in EC2 docker image
├── data_preprocess.py # Extracts and cleans JSON data
├── requirements.txt
├── Dockerfile
├── README.md
├── .github/  Future enhancements for ci/cd deployments
│ └── workflows/
│ └── deploy.yml # GitHub Actions for CI/CD
```


---


# 🧠 RAG Streamlit App Deployment on AWS EC2 with Docker

This guide walks you through deploying your Retrieval-Augmented Generation (RAG) app with Streamlit, ChromaDB, and image captioning on AWS EC2 using Docker.

---

## 📦 Prerequisites

- AWS account
- EC2 instance (Ubuntu recommended, t2.medium or higher)
- Port 8501 opened in the security group
- GitHub repository with your app (e.g., `rag-streamlit-app`)
- Docker installed on EC2
- AWS CLI configured (optional if uploading to S3)

---

## 🚀 1. Launch & Connect to EC2

- Launch EC2 instance from AWS Console
- Open **port 22** (SSH) and **8501** (for Streamlit)
- SSH into your EC2:

```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

---

## 🐳 2. Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

---

## 📁 3. Clone Your App

```bash
git clone https://github.com/<your-username>/rag-streamlit-app.git
cd rag-streamlit-app
```

---

## 📦 4. Build Docker Image

```bash
docker build -t rag-app .
```

---

## ▶️ 5. Run Docker Container

```bash
docker run -d -p 8501:8501 --name rag-container rag-app
```

---

## 🌐 6. Access the App

Visit in your browser:

```
http://<your-ec2-public-ip>:8501
```

---

## 🧹 7. Debugging

Check logs:

```bash
docker logs rag-container
```

Remove stale containers:

```bash
docker rm -f rag-container
```

Rebuild if needed:

```bash
docker build --no-cache -t rag-app .
docker run -d -p 8501:8501 --name rag-container rag-app
```

---

## 💾 8. Optional: Upload ChromaDB to S3

If enabled in your code:

```bash
aws s3 cp --recursive ./chroma_db s3://your-s3-bucket/chroma_db
```

---

## ✅ Done!

You're now running a fully functional RAG + Streamlit app with image captioning on EC2 using Docker.
