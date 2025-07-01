import boto3
import pandas as pd
import gzip
import json
import io

def get_clean_df_from_s3(bucket_name="your-bucket-name", prefix="ShopTalk/abo-llistings/"):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")


    # Use paginator to fetch all pages of .json.gz files
    gz_files = []
    print(f"🔍 Searching for .json.gz files in s3://{bucket_name}/{prefix} ...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json.gz"):
                gz_files.append(obj["Key"])

    print(f"📂 Found {len(gz_files)} gzipped JSON files.")

    data = []

    """for key in gz_files:
        print(f"📥 Reading file: {key}")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        bytestream = io.BytesIO(obj["Body"].read())

        with gzip.GzipFile(fileobj=bytestream, mode='rb') as f:
            for line in f:
                try:
                    decoded_line = line.decode('utf-8')
                    data.append(json.loads(decoded_line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping a line in {key} due to JSON error: {e}")"""

    for key in gz_files:
        print(f"📥 Reading file: {key}")

    # Fetch the object from S3
    print("📦 Fetching object from S3...")
    obj = s3.get_object(Bucket=bucket_name, Key=key)

    print("🔄 Reading byte stream from S3 object...")
    bytestream = io.BytesIO(obj["Body"].read())

    print("🗜️ Decompressing GZIP content...")
    with gzip.GzipFile(fileobj=bytestream, mode='rb') as f:
        print("📃 Starting line-by-line read and JSON decode...")
        line_number = 0
        for line in f:
            line_number += 1
            decoded_line = line.decode('utf-8')
            json_obj = json.loads(decoded_line)
            # Print progress every 1000 lines
            if line_number % 1000 == 0:
                print(f"✅ Processed {line_number} lines from {key}")

            data.append(json_obj)



    df = pd.DataFrame(data)
    print("=> Dataframe created")

    # Drop irrelevant columns
    drop_cols = [
        'item_weight', 'model_year', 'model_number', '3dmodel_id', 'finish_type', 'country',
        'marketplace', 'domain_name', 'node', 'spin_id', 'model_name', 'style',
        'fabric_type', 'pattern', 'finish_type', 'item_shape', 'item_dimensions'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace=True)

    # Extract 'en_US' from language-tagged fields
    def extract_en_us_value(data):
        if isinstance(data, list):
            values = [item['value'] for item in data if isinstance(item, dict) and item.get('language_tag') == 'en_US']
            return ' '.join(values)
        return None

    for col in ['brand', 'bullet_point', 'color', 'item_name', 'item_keywords', 'material', 'product_description']:
        if col in df.columns:
            df[col + '_en_US'] = df[col].apply(extract_en_us_value)
            df.drop(columns=col, inplace=True)

    # Flatten nested fields
    def extract_value_from_nested(data):
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'value' in data[0]:
                return data[0]['value']
            return data[0]
        elif isinstance(data, dict) and 'value' in data:
            return data['value']
        return data

    for col in ['product_type', 'color_code', 'other_image_id']:
        if col in df.columns:
            df[col] = df[col].apply(extract_value_from_nested)

    print("=> All columns are processed and removed not required columns")
    # Create text_blob
    def create_text_blob(row):
        parts = [
            str(row.get("item_name_en_US", "")),
            str(row.get("bullet_point_en_US", "")),
            str(row.get("item_keywords_en_US", "")),
            str(row.get("product_type", "")),
            str(row.get("brand_en_US", "")),
            str(row.get("color_en_US", "")),
            str(row.get("material_en_US", "")),
            str(row.get("product_description_en_US", ""))
        ]
        return " | ".join([p for p in parts if p and str(p).lower() not in ("nan", "none")])[:500]

    df["text_blob"] = df.apply(create_text_blob, axis=1)

    print("=> created text_blob column and added to Dataframe")
    # Final cleanup
    df = df[["item_id", "main_image_id", "other_image_id", "text_blob"]]
    df.dropna(subset=["item_id", "main_image_id", "text_blob"], inplace=True)

    print(f"✅ Clean DataFrame ready with {len(df)} rows.")
    return df