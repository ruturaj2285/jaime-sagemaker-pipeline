import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession

# --------------------------------------------------
# Basic config
# --------------------------------------------------
REGION = "ap-northeast-1"
PIPELINE_NAME = "poc-image-tag-pipeline2285"
ROLE_ARN = "arn:aws:iam::227295996532:role/sagemaker-service-role"
DEFAULT_BUCKET = "ml-demo-bucket2286"

# --------------------------------------------------
# Pick IMAGE_URI from exported *_IMAGE env vars
# --------------------------------------------------
PREPROCESS_IMAGE_URI = None
for key, value in os.environ.items():
    if key.endswith("_IMAGE"):
        PREPROCESS_IMAGE_URI = value
        print(f"Using image from env var: {key}")
        break

if not PREPROCESS_IMAGE_URI:
    raise RuntimeError("No *_IMAGE env var found")

print("Final PREPROCESS_IMAGE_URI:", PREPROCESS_IMAGE_URI)

# --------------------------------------------------
# SageMaker session (PIPELINE-AWARE)
# --------------------------------------------------
pipeline_session = PipelineSession(
    sagemaker_client=sagemaker.client("sagemaker", region_name=REGION),
    default_bucket=DEFAULT_BUCKET,
)

# --------------------------------------------------
# Pipeline parameters
# --------------------------------------------------
input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{DEFAULT_BUCKET}/data/input.csv",
)

# --------------------------------------------------
# Preprocessing step (custom Docker image)
# --------------------------------------------------
processor = Processor(
    image_uri=PREPROCESS_IMAGE_URI,
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session,
)

step_process = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/output",
        )
    ],
)

# --------------------------------------------------
# Training step (managed XGBoost)
# --------------------------------------------------
TRAIN_IMAGE_URI = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.5-1",
)

estimator = Estimator(
    image_uri=TRAIN_IMAGE_URI,
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{DEFAULT_BUCKET}/training-output/",
    sagemaker_session=pipeline_session,
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": step_process.properties
        .ProcessingOutputConfig.Outputs["train_data"]
        .S3Output.S3Uri
    },
)

# --------------------------------------------------
# Register model
# --------------------------------------------------
step_register = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="poc-image-tag-model-group",
    approval_status="PendingManualApproval",
)

# --------------------------------------------------
# Pipeline definition
# --------------------------------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
    sagemaker_session=pipeline_session,
)

# --------------------------------------------------
# Upsert pipeline (CREATE or UPDATE)
# --------------------------------------------------
if __name__ == "__main__":
    print("🔄 Upserting SageMaker pipeline...")
    pipeline.upsert(role_arn=ROLE_ARN)

    details = pipeline.describe()
    print("✅ Pipeline upserted successfully")
    print("🔗 Pipeline ARN:", details["PipelineArn"])
