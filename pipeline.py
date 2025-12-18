import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.estimator import Estimator

# ------------------------------------------------
# Config
# ------------------------------------------------
REGION = "ap-northeast-1"
PIPELINE_NAME = "poc-image-tag-pipeline"

# ------------------------------------------------
# Find IMAGE_URI from exported *_IMAGE env vars
# ------------------------------------------------
image_env_vars = [
    "mdl-data-collection_IMAGE",
    "mdl-pre-processing_IMAGE",
    "mdl-feature-correlation_IMAGE",
    "mdl-feature-importance_IMAGE",
    "mdl-training_IMAGE",
]

IMAGE_URI = None
for var in image_env_vars:
    if os.getenv(var):
        IMAGE_URI = os.getenv(var)
        print(f"Using image from env var: {var}")
        break

if not IMAGE_URI:
    raise RuntimeError(
        "No image env var found. Expected one of: "
        + ", ".join(image_env_vars)
    )

print("Final IMAGE_URI:", IMAGE_URI)

# ------------------------------------------------
# SageMaker session & role
# ------------------------------------------------
session = sagemaker.Session(
    boto3.Session(region_name=REGION)
)

role = sagemaker.get_execution_role(session)

# ------------------------------------------------
# Dummy TrainingStep (pipeline will NOT run)
# ------------------------------------------------
estimator = Estimator(
    image_uri=IMAGE_URI,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
)

train_step = TrainingStep(
    name="ImageTagVerificationStep",
    estimator=estimator,
)

# ------------------------------------------------
# Pipeline definition
# ------------------------------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[train_step],
    sagemaker_session=session,
)

# ------------------------------------------------
# Create / Update pipeline ONLY
# ------------------------------------------------
if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully")
