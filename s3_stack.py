import boto3

# using LocalStack
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

bucket_name = "my-bucket"
model_file_path = "artifact/experiment/Production/xgboost.pkl"

# Create bucket
s3.create_bucket(Bucket=bucket_name)

# Upload model file to S3
s3.upload_file(model_file_path, bucket_name, "s3_model.pkl")

print(f"Model uploaded to S3 bucket {bucket_name} with key 's3_model.pkl'")