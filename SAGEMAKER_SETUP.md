# AWS SageMaker Training Setup Guide

This guide explains how to run your vessel segmentation training on AWS SageMaker instead of your local machine.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials
3. **Docker** installed on your local machine
4. **Python 3.8+** with boto3 and sagemaker SDK

## Step-by-Step Setup

### Step 1: Install Required Tools

```bash
# Install AWS CLI (if not already installed)
# Windows: Download from https://aws.amazon.com/cli/
# macOS: brew install awscli
# Linux: sudo apt-get install awscli

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Key, and default region

# Install SageMaker SDK
pip install sagemaker boto3
```

### Step 2: Upload Your Augmentation Config to S3

```bash
# Upload the default config to S3
aws s3 cp configs/default_config.json s3://vessel-segmentation-data/configs/default_config.json

# Verify upload
aws s3 ls s3://vessel-segmentation-data/configs/
```

### Step 3: Build and Push Docker Container

The training code needs to be packaged in a Docker container and pushed to Amazon ECR (Elastic Container Registry).

```bash
# Make the script executable (Linux/macOS)
chmod +x build_and_push.sh

# Build and push the container
# Usage: ./build_and_push.sh [region] [repository-name] [tag]
./build_and_push.sh us-east-1 hyperion-unet-training latest
```

**For Windows users**, you can run the commands manually:

```powershell
# Get your AWS account ID
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$REGION = "us-east-1"
$REPOSITORY = "hyperion-unet-training"

# Create ECR repository
aws ecr create-repository --repository-name $REPOSITORY --region $REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Build the Docker image
docker build -t ${REPOSITORY}:latest .

# Tag the image
docker tag ${REPOSITORY}:latest "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${REPOSITORY}:latest"

# Push to ECR
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${REPOSITORY}:latest"
```

### Step 4: Set Up IAM Role for SageMaker

Your SageMaker execution role needs the following permissions:

**Required IAM Policies:**
1. `AmazonSageMakerFullAccess` (managed policy)
2. Custom policy for S3 access:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::vessel-segmentation-data/*",
                "arn:aws:s3:::vessel-segmentation-data"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

**To create/update the role:**
1. Go to AWS IAM Console → Roles
2. Create a new role named `SageMaker-UNetTraining-ExecutionRole`
3. Attach the policies above
4. Note the role ARN (you may need it)

### Step 5: Submit a Training Job

Now you're ready to submit a training job!

```bash
# Basic training job (10 epochs, default settings)
python sagemaker_train.py --epochs 10 --batch-size 8

# Production training job (GPU instance, 50 epochs)
python sagemaker_train.py \
    --instance-type ml.g4dn.xlarge \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3

# Cost-optimized training (using spot instances - up to 70% cheaper!)
python sagemaker_train.py \
    --instance-type ml.g4dn.xlarge \
    --epochs 50 \
    --use-spot

# Custom job name and config
python sagemaker_train.py \
    --job-name my-vessel-segmentation-experiment \
    --config-s3-uri s3://vessel-segmentation-data/configs/my_custom_config.json \
    --epochs 100 \
    --batch-size 8
```

### Step 6: Monitor Your Training Job

**Option 1: AWS Console**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs
```

**Option 2: AWS CLI**
```bash
# List recent jobs
aws sagemaker list-training-jobs --max-results 5

# Describe specific job
aws sagemaker describe-training-job --training-job-name vessel-segmentation-2025-10-19-14-30-00

# Stream logs in real-time
aws logs tail /aws/sagemaker/TrainingJobs --follow --format short --filter-pattern 'vessel-segmentation'
```

**Option 3: Python SDK**
```python
import sagemaker
session = sagemaker.Session()

# Attach to existing job
estimator = sagemaker.estimator.Estimator.attach('vessel-segmentation-2025-10-19-14-30-00')

# Get logs
estimator.logs()
```

## SageMaker Instance Types & Pricing

| Instance Type | vCPUs | GPU | Memory | Price/hour (approx) | Use Case |
|---------------|-------|-----|--------|---------------------|----------|
| ml.g4dn.xlarge | 4 | 1x T4 (16GB) | 16 GB | $0.736 | Development/Testing |
| ml.g4dn.2xlarge | 8 | 1x T4 (16GB) | 32 GB | $0.94 | Small-medium datasets |
| ml.p3.2xlarge | 8 | 1x V100 (16GB) | 61 GB | $3.825 | Production training |
| ml.p3.8xlarge | 32 | 4x V100 (64GB) | 244 GB | $14.688 | Large-scale training |

**Spot Instance Savings:** Use `--use-spot` flag to save up to 70% (training may be interrupted but will resume from checkpoint)

## Understanding SageMaker Data Channels

When SageMaker starts your training job, it:

1. **Downloads data from S3** to the instance at `/opt/ml/input/data/<channel_name>`
   - `train` channel → `/opt/ml/input/data/train/`
   - `val` channel → `/opt/ml/input/data/val/`
   - `config` channel → `/opt/ml/input/data/config/`

2. **Sets environment variables:**
   - `SM_CHANNEL_TRAIN=/opt/ml/input/data/train`
   - `SM_CHANNEL_VAL=/opt/ml/input/data/val`
   - `SM_MODEL_DIR=/opt/ml/model`

3. **Your train.py script** uses these environment variables (already implemented!)

4. **After training completes**, SageMaker uploads everything in `/opt/ml/model` to S3

## Modifying Training for SageMaker

### Config File Handling

You need to update `train.py` to handle the config file from the SageMaker channel:

```python
# Add this to parse_args() in train.py
p.add_argument("--config-file", type=str,
               default=os.environ.get("SM_CHANNEL_CONFIG", None))

# In main(), update the config file path
config_file = args.config_file
if config_file and os.path.isdir(config_file):
    # SageMaker downloads channel to a directory
    config_file = os.path.join(config_file, 'default_config.json')
```

I'll create a patch for this below.

## Troubleshooting

### Issue: "Could not find image"
**Solution:** Make sure you ran `build_and_push.sh` successfully and the image is in ECR

### Issue: "Access Denied" when accessing S3
**Solution:** Check your SageMaker execution role has permissions for the S3 bucket

### Issue: "ResourceLimitExceeded"
**Solution:** You've hit your SageMaker instance limit. Request a quota increase in AWS Service Quotas

### Issue: Training job stuck or slow
**Solution:**
- Check CloudWatch logs for errors
- Verify data is accessible in S3
- Ensure instance type has sufficient GPU memory for your batch size

### Issue: Out of Memory (OOM)
**Solution:**
- Reduce batch size (`--batch-size 4` or `--batch-size 2`)
- Use a larger instance type (e.g., ml.g4dn.2xlarge)
- Check if data augmentation is loading too much into memory

## Cost Optimization Tips

1. **Use spot instances** (`--use-spot`) - saves up to 70%
2. **Start small** - test with ml.g4dn.xlarge before scaling
3. **Set max run time** appropriately - don't pay for stuck jobs
4. **Delete old checkpoints** - clean up S3 to reduce storage costs
5. **Use cheaper instance for validation** - only need GPU for training

## Next Steps

After your training completes:

1. **Download model artifacts:**
   ```bash
   aws s3 cp s3://vessel-segmentation-data/experiments/models/vessel-segmentation-2025-10-19/output/model.tar.gz .
   tar -xzf model.tar.gz
   ```

2. **Evaluate on test set** (implement in `testing/` directory)

3. **Deploy model** using SageMaker Endpoints for inference

4. **Track experiments** - consider integrating MLflow or Weights & Biases

## Files Created for SageMaker

- `Dockerfile` - Container definition with all dependencies
- `build_and_push.sh` - Script to build and push Docker image to ECR
- `sagemaker_train.py` - Python script to submit training jobs
- `requirements.txt` - Python dependencies (already created)
- `configs/default_config.json` - Example augmentation config (already created)

## Quick Start Checklist

- [ ] AWS CLI installed and configured
- [ ] Docker installed
- [ ] Config uploaded to S3: `aws s3 cp configs/default_config.json s3://vessel-segmentation-data/configs/`
- [ ] Docker image built and pushed: `./build_and_push.sh`
- [ ] IAM role configured with proper permissions
- [ ] Submit training job: `python sagemaker_train.py --epochs 10`
- [ ] Monitor job in AWS Console or via logs

Good luck with your SageMaker training! 🚀
