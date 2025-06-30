import boto3
import os

def download_chroma_from_s3(bucket="your-bucket", prefix="chroma_db/", local_dir="/tmp/chroma_db"):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix):]
            if not rel_path:
                continue
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

if __name__ == "__main__":
    download_chroma_from_s3()
