# prepare_index.py

import os
import io
import boto3
import torch
import gzip
import pandas as pd
from PIL import Image
from io import BytesIO

from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from data_preprocess import get_clean_df_from_s3

# Constants
BUCKET_NAME = "shubendu-rag-llm-app-bucket"
CHROMA_PERSIST_PATH = "./chroma_db" # local chromadb path in EC2
IMAGE_METADATA_KEY_PATH = "ShopTalk/abo-images-small/metadata/images.csv.gz"

# S3 Client
s3 = boto3.client("s3")

# Get the gzipped file from S3
response = s3.get_object(Bucket=BUCKET_NAME, Key=IMAGE_METADATA_KEY_PATH)
with gzip.GzipFile(fileobj=BytesIO(response['Body'].read())) as gz:
    image_meta_df = pd.read_csv(gz)


#Creates a Python dictionary: image_id → s3 path
image_id_to_path = dict(zip(image_meta_df["image_id"].str.lower(), image_meta_df["path"]))

def get_image_by_id_from_metadata(image_id):
    """Retrieve image directly from S3 using metadata path."""
    key = image_id_to_path.get(image_id.lower())
    if not key:
        return None, None
    try:
        key = f"ShopTalk/abo-images-small/small/{key}"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB"), key
    except Exception as e:
        print(f"[❌ Failed] Could not fetch {key} from S3: {e}")
        return None, key

def prepare_index():
    df = get_clean_df_from_s3(BUCKET_NAME)[:100]
    df.head(20)

    # Load models
    #Smaller models
    '''embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
    '''

    # Larger models
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    '''caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16
    )'''
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"
                                                                 ).to("cuda" if torch.cuda.is_available() else "cpu")

    documents = []
    print("📁 Starting document creation and image captioning...")

    print(f"📁 Going through each df row and doing captioning")
    for idx, row in enumerate(df.iterrows(), 1):  # Start index from 1
        _, row = row
        item_id = str(row["item_id"])
        text_blob = str(row["text_blob"])
        main_image_id = str(row["main_image_id"]).lower()

        # Fetch and caption image
        img, s3_key = get_image_by_id_from_metadata(main_image_id)

        if img:
            try:
                inputs = caption_processor(img, return_tensors="pt").to(caption_model.device)
                out = caption_model.generate(**inputs)
                caption = caption_processor.decode(out[0], skip_special_tokens=True)
                image_path = f"s3://{BUCKET_NAME}/{s3_key}"
            except Exception as e:
                print(f"[BLIP error] Failed to caption {main_image_id}: {e}")
                caption = ""
                image_path = None
        else:
            caption = ""
            image_path = None

        # Merge text_blob and caption
        combined_text = f"{text_blob} \n Image Caption -  {caption}" if caption else text_blob

        documents.append(Document(
            page_content=combined_text,
            metadata={
                "item_id": item_id,
                "image_path": image_path
            }
        ))

        # ✅ Print progress every 5,000 items
        if idx % 5 == 0:
            print(f"✅ Processed {idx} rows")

    print("📁 Building Chroma index locally")
    vectordb = Chroma.from_documents(
        documents,
        embedding_function,
        persist_directory=CHROMA_PERSIST_PATH
    )

    # Optional: remove if using ChromaDB >= 0.4.x
    vectordb.persist()

    print(f"\n✅ ChromaDB index created with {len(documents)} documents.")
    print(f"📁 Local Chroma stored at: {CHROMA_PERSIST_PATH}")

if __name__ == "__main__":
    prepare_index()
