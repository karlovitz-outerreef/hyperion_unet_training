#!/bin/bash

# Build and push Docker image to AWS ECR for SageMaker training
# Usage: ./build_and_push.sh [region] [repository-name]

set -e

# Configuration
REGION=${1:-us-east-1}
REPOSITORY_NAME=${2:-hyperion-unet-training}
IMAGE_TAG=${3:-latest}

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$ACCOUNT_ID" ]; then
    echo "Error: Could not get AWS account ID. Make sure AWS CLI is configured."
    exit 1
fi

echo "=== Building and Pushing Docker Image ==="
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "Repository: $REPOSITORY_NAME"
echo "Image Tag: $IMAGE_TAG"
echo ""

# Full image name
FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

# Create ECR repository if it doesn't exist
echo "=== Creating ECR repository (if needed) ==="
aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --region ${REGION} > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region ${REGION}

echo "Repository: ${REPOSITORY_NAME} is ready"

# Authenticate Docker to ECR
echo ""
echo "=== Authenticating Docker to ECR ==="
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build Docker image
echo ""
echo "=== Building Docker image ==="
docker build -t ${REPOSITORY_NAME}:${IMAGE_TAG} .

# Tag image for ECR
echo ""
echo "=== Tagging image ==="
docker tag ${REPOSITORY_NAME}:${IMAGE_TAG} ${FULLNAME}

# Push to ECR
echo ""
echo "=== Pushing image to ECR ==="
docker push ${FULLNAME}

echo ""
echo "=== Build and Push Complete ==="
echo "Image URI: ${FULLNAME}"
echo ""
echo "You can now submit a training job using:"
echo "  python sagemaker_train.py --instance-type ml.g4dn.xlarge --epochs 50"
