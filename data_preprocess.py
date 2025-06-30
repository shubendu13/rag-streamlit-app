# data_preprocess.py

import zipfile
import json
import pandas as pd
import os
import gzip
from google.colab import drive

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

def get_clean_df():
    # Mount Google Drive
    drive.mount('/content/drive')

    # Path to JSON files in Drive
    directory_path = "/content/drive/My Drive/Colab Notebooks/ShopTalkData/abo-listings/listings/metadata/"

    # List all .json.gz files
    gz_files = [f for f in os.listdir(directory_path) if f.endswith('.json.gz')]
    data = []
    print(f"Found {len(gz_files)} gzipped JSON files.")

    # Read and parse all files
    for file_name in gz_files:
        file_path = os.path.join(directory_path, file_name)
        print(f"Reading file: {file_name}")
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping line {line_num + 1} in {file_name} due to JSONDecodeError: {e}")
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

    print(f"Finished reading {len(data)} records from gzipped JSON files.")

    # Flattening
    df = pd.DataFrame(data).drop(columns=[
        'item_weight', 'model_year', 'model_number', '3dmodel_id', 'finish_type', 'country',
        'marketplace', 'domain_name', 'node', 'spin_id', 'model_name', 'style', 'fabric_type',
        'pattern', 'finish_type', 'item_shape', 'item_dimensions'
    ])

    # Extract 'en_US' values
    def extract_en_us_value(data):
        if isinstance(data, list):
            values = [item['value'] for item in data if isinstance(item, dict) and item.get('language_tag') == 'en_US' and 'value' in item]
            return ' '.join(values)
        return None

    columns_to_process_lang = ['brand', 'bullet_point', 'color', 'item_name', 'item_keywords', 'material', 'fabric_type', 'product_description']
    for col in columns_to_process_lang:
        if col in df.columns:
            df[col + '_en_US'] = df[col].apply(extract_en_us_value)
    df = df.drop(columns=columns_to_process_lang, errors='ignore')

    # Extract values from nested
    def extract_value_from_nested(data):
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'value' in data[0]:
                return data[0]['value']
            return data[0]
        elif isinstance(data, dict) and 'value' in data:
            return data['value']
        return data

    columns_with_nested = ['product_type', 'color_code', 'other_image_id']
    for col in columns_with_nested:
        if col in df.columns:
            df[col] = df[col].apply(extract_value_from_nested)

    # Create text_blob
    max_text_blob_length = 500

    def create_text_blob(row):
        parts = [
            str(row.get("item_name_en_US", "")),
            str(row.get("bullet_point_en_US", "")),
            str(row.get("item_keywords_en_US", "")),
            str(row.get("product_type", "")),
            str(row.get("brand_en_US", "")),
            str(row.get("color_en_US")),
            str(row.get("material_en_US", "")),
            str(row.get("product_description_en_US", ""))
        ]
        return " | ".join([p for p in parts if p and str(p).lower() not in ("nan", "none")])[:max_text_blob_length]

    df["text_blob"] = df.apply(create_text_blob, axis=1)

    cols_to_drop = [
        "item_name_en_US", "bullet_point_en_US", "item_keywords_en_US", "product_type",
        "brand_en_US", "color_en_US", "color_code", "material_en_US", "product_description_en_US"
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    return df
