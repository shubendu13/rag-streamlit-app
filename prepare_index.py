# prepare_index.py (S3-compatible version)

import os
import io
import boto3
import torch
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from data_preprocess import get_clean_df_from_s3  # make sure this exists

# Constants
BUCKET_NAME = "shubendu-rag-llm-app-bucket"
IMAGE_PREFIX = "ShopTalk/abo-images-small/images/small/"
CHROMA_PERSIST_PATH = "/tmp/chroma_db"  # local build path (then upload to S3)

# S3 client
s3 = boto3.client("s3")


def download_image_from_s3(main_image_id):
    """Searches and downloads image by image_id from S3."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=IMAGE_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if main_image_id.lower() in key.lower():
                try:
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    image_bytes = response["Body"].read()
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), key
                except Exception as e:
                    print(f"Failed to read image {key}: {e}")
    return None, None


def prepare_index():
    df = get_clean_df_from_s3(BUCKET_NAME)

    # Load models
    #Smaller models
    '''embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
    '''

    # Larger models
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    documents = []

    for _, row in df.iterrows():
        item_id = str(row["item_id"])
        text_blob = str(row["text_blob"])
        main_image_id = str(row["main_image_id"]).lower()

        # Add text entry
        documents.append(Document(
            page_content=text_blob,
            metadata={"item_id": item_id, "type": "text", "image_path": None}
        ))

        # Add image caption
        img, s3_key = download_image_from_s3(main_image_id)
        if img:
            try:
                inputs = caption_processor(img, return_tensors="pt").to(caption_model.device)
                out = caption_model.generate(**inputs)
                caption = caption_processor.decode(out[0], skip_special_tokens=True)

                documents.append(Document(
                    page_content=caption,
                    metadata={"item_id": item_id, "type": "image", "image_path": f"s3://{BUCKET_NAME}/{s3_key}"}
                ))
            except Exception as e:
                print(f"[BLIP caption error] {s3_key}: {e}")

    # Build ChromaDB index locally
    vectordb = Chroma.from_documents(
        documents, embedding_function, persist_directory=CHROMA_PERSIST_PATH
    )
    vectordb.persist()

    print(f"\n✅ ChromaDB index created with {len(documents)} documents.")
    print(f"📁 Local Chroma stored at: {CHROMA_PERSIST_PATH}")


if __name__ == "__main__":
    prepare_index()
