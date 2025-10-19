# SageMaker Quick Start - TL;DR

Run your training on AWS SageMaker in 5 steps:

## 1. Prerequisites
```bash
pip install sagemaker boto3
aws configure  # Enter your AWS credentials
```

## 2. Upload Config to S3
```bash
aws s3 cp configs/default_config.json s3://vessel-segmentation-data/configs/
```

## 3. Build & Push Docker Container
```bash
# Linux/macOS
chmod +x build_and_push.sh
./build_and_push.sh

# Windows PowerShell
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$REGION = "us-east-1"
$REPO = "hyperion-unet-training"

aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -t ${REPO}:latest .
docker tag ${REPO}:latest "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${REPO}:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${REPO}:latest"
```

## 4. Submit Training Job
```bash
# Basic job
python sagemaker_train.py --epochs 50 --batch-size 8

# Production job with spot instances (70% cheaper!)
python sagemaker_train.py \
    --instance-type ml.g4dn.xlarge \
    --epochs 50 \
    --batch-size 16 \
    --use-spot
```

## 5. Monitor Training
```bash
# View in AWS Console
https://console.aws.amazon.com/sagemaker/home#/jobs

# Or stream logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern 'vessel-segmentation'
```

## Common Instance Types
- `ml.g4dn.xlarge` - $0.74/hr - 1x T4 GPU (good for most cases)
- `ml.p3.2xlarge` - $3.82/hr - 1x V100 GPU (faster training)

**Tip:** Add `--use-spot` to save up to 70%!

## Troubleshooting
- **"Access Denied"**: Check IAM role has S3 permissions
- **"Image not found"**: Run `build_and_push.sh` again
- **OOM error**: Reduce `--batch-size` to 4 or 2

For detailed documentation, see `SAGEMAKER_SETUP.md`
