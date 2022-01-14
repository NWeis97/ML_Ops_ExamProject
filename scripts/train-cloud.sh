#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs cloud training for a PyTorch model.
echo "Training cloud ML model"

# PROJECT_ID: project id on gcp
PROJECT_ID=examproject-mlops

# BUCKET_ID: bucket id for saving trained model
BUCKET_ID=gpt2_exam_project_bucket

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=dockers

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=latest

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=examproject_mlops$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=europe-west1


# Build the docker image
# docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
# docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/models/base

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier CUSTOM \
    --master-machine-type n1-highmem-8 \
    -- \
    --job-dir=${JOB_DIR} \
    --project-id=${PROJECT_ID} \
    -- \
    --WANDB_API_KEY=""  #Set own WANDB_API_KEY

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*
