import boto3


def localstack_s3():
    # Using LocalStack
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )

    bucket_name = "risk-bucket"
    model_file_path = "artifact/experiment/Production/xgboost.pkl"
    processor_path = "artifact/transform/transformer.pkl"

    # Create bucket
    s3.create_bucket(Bucket=bucket_name)

    # Upload model file to S3
    s3.upload_file(model_file_path, bucket_name, "s3_risk_model.pkl")

    # Upload processor file to S3
    s3.upload_file(processor_path, bucket_name, "s3_transformer.pkl")

    print(f"Model and processor uploaded to S3 bucket {bucket_name}")
