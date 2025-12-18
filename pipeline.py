import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.estimator import Estimator

# -----------------------------
# Basic config
# -----------------------------
REGION = "ap-northeast-1"
PIPELINE_NAME = "poc-image-tag-pipeline"

# -----------------------------
# Read image URI from env
# -----------------------------
IMAGE_URI = os.environ.get("IMAGE_URI")

if not IMAGE_URI:
    raise ValueError("IMAGE_URI environment variable not set")

print("Using IMAGE_URI:", IMAGE_URI)

# -----------------------------
# SageMaker session & role
# -----------------------------
session = sagemaker.Session(
    boto3.Session(region_name=REGION)
)

role = sagemaker.get_execution_role(session)

# -----------------------------
# Dummy training step
# (pipeline will NOT be executed)
# -----------------------------
estimator = Estimator(
    image_uri=IMAGE_URI,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
)

train_step = TrainingStep(
    name="DummyTrainingStep",
    estimator=estimator,
)

# -----------------------------
# Pipeline definition
# -----------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[train_step],
    sagemaker_session=session,
)

# -----------------------------
# Create / Update pipeline ONLY
# -----------------------------
if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully")
