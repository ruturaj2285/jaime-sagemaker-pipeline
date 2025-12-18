import os
import json
import boto3

REGION = "ap-northeast-1"
PIPELINE_NAME = "poc-image-tag-pipeline"

# --------------------------------------------------
# Pick IMAGE_URI from exported *_IMAGE env vars
# --------------------------------------------------
IMAGE_URI = None
for key, value in os.environ.items():
    if key.endswith("_IMAGE"):
        IMAGE_URI = value
        print(f"Using image from env var: {key}")
        break

if not IMAGE_URI:
    raise RuntimeError("No *_IMAGE env var found")

print("Final IMAGE_URI:", IMAGE_URI)

# --------------------------------------------------
# AWS clients
# --------------------------------------------------
sm = boto3.client("sagemaker", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)

# --------------------------------------------------
# SageMaker execution role
# --------------------------------------------------
EXECUTION_ROLE_NAME = "sagemaker-execution-role"  # change if needed

execution_role_arn = iam.get_role(
    RoleName=EXECUTION_ROLE_NAME
)["Role"]["Arn"]

print("Using execution role:", execution_role_arn)

# --------------------------------------------------
# Minimal pipeline definition
# --------------------------------------------------
pipeline_definition = {
    "Version": "2020-12-01",
    "Steps": [
        {
            "Name": "ImageTagCheckStep",
            "Type": "Training",
            "Arguments": {
                "AlgorithmSpecification": {
                    "TrainingImage": IMAGE_URI,
                    "TrainingInputMode": "File"
                },
                "RoleArn": execution_role_arn,
                "OutputDataConfig": {
                    "S3OutputPath": "s3://dummy-bucket-do-not-run/"
                },
                "ResourceConfig": {
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 1
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 60
                }
            }
        }
    ]
}

# --------------------------------------------------
# Create or Update pipeline
# --------------------------------------------------
created = False

try:
    resp = sm.describe_pipeline(PipelineName=PIPELINE_NAME)
    pipeline_arn = resp["PipelineArn"]
    print("Pipeline exists → updating")
    sm.update_pipeline(
        PipelineName=PIPELINE_NAME,
        PipelineDefinition=json.dumps(pipeline_definition),
        RoleArn=execution_role_arn
    )
except sm.exceptions.ResourceNotFound:
    print("Pipeline does not exist → creating")
    resp = sm.create_pipeline(
        PipelineName=PIPELINE_NAME,
        PipelineDefinition=json.dumps(pipeline_definition),
        RoleArn=execution_role_arn
    )
    pipeline_arn = resp["PipelineArn"]
    created = True

# --------------------------------------------------
# Print Pipeline ARN (FINAL OUTPUT)
# --------------------------------------------------
print("--------------------------------------------------")
print("SageMaker Pipeline ARN:")
print(pipeline_arn)
print("Status:", "CREATED" if created else "UPDATED")
print("--------------------------------------------------")
