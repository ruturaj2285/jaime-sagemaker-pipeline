import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator

REGION = "ap-northeast-1"
PIPELINE_NAME = "mdl-ml-pipeline"

session = sagemaker.Session(boto3.Session(region_name=REGION))
role = sagemaker.get_execution_role(session)

ecr = boto3.client("ecr", region_name=REGION)
account_id = boto3.client("sts").get_caller_identity()["Account"]

def latest_image(repo):
    images = ecr.describe_images(repositoryName=repo)["imageDetails"]
    images.sort(key=lambda x: x["imagePushedAt"], reverse=True)
    tag = images[0]["imageTags"][0]
    return f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{repo}:{tag}"

DATA_COLLECTION_IMAGE = os.getenv("mdl-data-collection_IMAGE")     or latest_image("mdl-data-collection")
PREPROCESS_IMAGE      = os.getenv("mdl-pre-processing_IMAGE")      or latest_image("mdl-pre-processing")
FEATURE_CORR_IMAGE    = os.getenv("mdl-feature-correlation_IMAGE") or latest_image("mdl-feature-correlation")
FEATURE_IMPORT_IMAGE  = os.getenv("mdl-feature-importance_IMAGE")  or latest_image("mdl-feature-importance")
TRAIN_IMAGE           = os.getenv("mdl-training_IMAGE")            or latest_image("mdl-training")

print("Using images:")
print("Data Collection     :", DATA_COLLECTION_IMAGE)
print("Pre Processing      :", PREPROCESS_IMAGE)
print("Feature Correlation :", FEATURE_CORR_IMAGE)
print("Feature Importance  :", FEATURE_IMPORT_IMAGE)
print("Training            :", TRAIN_IMAGE)

data_collection = ProcessingStep(
    name="DataCollection",
    processor=ScriptProcessor(
        image_uri=DATA_COLLECTION_IMAGE,
        command=["python3", "data_collection.py"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
    ),
)

preprocess = ProcessingStep(
    name="PreProcessing",
    processor=ScriptProcessor(
        image_uri=PREPROCESS_IMAGE,
        command=["python3", "preprocessing.py"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
    ),
)

feature_corr = ProcessingStep(
    name="FeatureCorrelation",
    processor=ScriptProcessor(
        image_uri=FEATURE_CORR_IMAGE,
        command=["python3", "feature_correlation.py"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
    ),
)

feature_importance = ProcessingStep(
    name="FeatureImportance",
    processor=ScriptProcessor(
        image_uri=FEATURE_IMPORT_IMAGE,
        command=["python3", "feature_importance.py"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
    ),
)

training = TrainingStep(
    name="Training",
    estimator=Estimator(
        image_uri=TRAIN_IMAGE,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        entry_point="train.py",
    ),
)

pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[
        data_collection,
        preprocess,
        feature_corr,
        feature_importance,
        training,
    ],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    print(f"SageMaker pipeline '{PIPELINE_NAME}' updated successfully")
