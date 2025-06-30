# utils/upload_chroma.py
import boto3
import os

def upload_chroma_db_to_s3(local_path, bucket_name, s3_prefix):
    """
    Recursively upload all files from a local directory to an S3 bucket.
    """
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Windows-safe

            try:
                s3.upload_file(local_file_path, bucket_name, s3_key)
                print(f"✅ Uploaded: {s3_key}")
            except Exception as e:
                print(f"❌ Failed to upload {s3_key}: {e}")
