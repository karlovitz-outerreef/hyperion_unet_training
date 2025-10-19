#!/usr/bin/env python3
"""
SageMaker Training Job Submission Script

This script builds a Docker container and submits a training job to AWS SageMaker.
"""

import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from datetime import datetime
import os


def submit_training_job(
    job_name=None,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    volume_size=50,
    max_run_time=86400,  # 24 hours in seconds
    epochs=10,
    batch_size=8,
    learning_rate=1e-3,
    config_s3_uri=None,
    use_spot=False,
    job_comment=None,
):
    """
    Submit a SageMaker training job.

    Args:
        job_name: Name for the training job (default: auto-generated)
        instance_type: SageMaker instance type (e.g., ml.g4dn.xlarge, ml.p3.2xlarge)
        instance_count: Number of instances
        volume_size: EBS volume size in GB
        max_run_time: Maximum training time in seconds
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        config_s3_uri: S3 URI to augmentation config (e.g., s3://bucket/path/config.json)
        use_spot: Use spot instances for cost savings
    """

    # Setup
    session = sagemaker.Session()
    region = session.boto_region_name
    account_id = boto3.client('sts').get_caller_identity()['Account']

    # Get SageMaker execution role
    # Option 1: Use environment variable if set
    # Option 2: Try to get role from SageMaker notebook (if running in notebook)
    # Option 3: Use manually specified role ARN
    role = os.environ.get('SAGEMAKER_ROLE')
    if not role:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            # Not running in SageMaker notebook, need to specify role manually
            role = f'arn:aws:iam::{account_id}:role/SageMaker-UNetTraining-ExecutionRole'
            print(f"Using default role: {role}")
            print(f"If this role doesn't exist, create it or set SAGEMAKER_ROLE environment variable")

    # Generate job name if not provided
    if job_name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        job_name = f'vessel-segmentation-{timestamp}'

    # Docker image details
    ecr_repository = 'hyperion-unet-training'
    image_tag = 'latest'
    image_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}:{image_tag}'

    print(f"Using Docker image: {image_uri}")

    # S3 paths
    bucket = 'vessel-segmentation-data'
    train_data = f's3://{bucket}/data/processed/train'
    val_data = f's3://{bucket}/data/processed/val'
    output_path = f's3://{bucket}/experiments/models/{job_name}'
    checkpoint_path = f's3://{bucket}/experiments/checkpoints/{job_name}'
    log_path = f's3://{bucket}/experiments/logs/{job_name}'

    # Default config if not provided
    if config_s3_uri is None:
        config_s3_uri = f's3://{bucket}/configs/default_config.json'

    print(f"\n=== Training Job Configuration ===")
    print(f"Job Name: {job_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Training Data: {train_data}")
    print(f"Validation Data: {val_data}")
    print(f"Config: {config_s3_uri}")
    print(f"Output Path: {output_path}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Spot Instances: {use_spot}")

    # Hyperparameters passed to training script
    hyperparameters = {
        'epochs': epochs,
        'batch-size': batch_size,
        'lr': learning_rate,
        'images-path': 'images',
        'labels-path': 'labels',
        'num-workers': 4,  # Adjust based on instance vCPUs
        'seed': 42,
    }

    # Add job comment if provided
    if job_comment:
        hyperparameters['job-comment'] = job_comment

    # Data channels for SageMaker
    data_channels = {
        'train': train_data,
        'val': val_data,
        'config': config_s3_uri,  # Config file as a channel
    }

    # Metric definitions for SageMaker to parse from logs
    metric_definitions = [
        {'Name': 'train:dice', 'Regex': r'train dice=([0-9.]+)'},
        {'Name': 'train:ce', 'Regex': r'ce=([0-9.]+)'},
        {'Name': 'val:dice', 'Regex': r'val dice=([0-9.]+)'},
        {'Name': 'val:ce', 'Regex': r'val ce=([0-9.]+)'},
    ]

    # Spot instance configuration
    spot_config = {}
    if use_spot:
        spot_config = {
            'use_spot_instances': True,
            'max_wait': max_run_time + 3600,  # Add 1 hour buffer
            'checkpoint_s3_uri': checkpoint_path,
        }

    # Create estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=volume_size,
        max_run=max_run_time,
        output_path=output_path,
        base_job_name='vessel-segmentation',
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        environment={
            'PYTHONUNBUFFERED': '1',  # For real-time logging
            'TRAINING_JOB_NAME': job_name,  # For logging
        },
        **spot_config
    )

    print(f"\n=== Submitting Training Job ===")
    print(f"Data channels being passed:")
    for channel_name, s3_uri in data_channels.items():
        print(f"  {channel_name}: {s3_uri}")
    print(f"This may take a few minutes to start...")

    # Start training
    estimator.fit(
        inputs=data_channels,
        job_name=job_name,
        wait=False,  # Set to True if you want to wait for completion
        logs='All',
    )

    print(f"\n=== Training Job Submitted ===")
    print(f"Job Name: {job_name}")
    print(f"\nMonitor your job at:")
    print(f"https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    print(f"\nTo view logs:")
    print(f"aws logs tail /aws/sagemaker/TrainingJobs --follow --format short --filter-pattern '{job_name}'")
    print(f"\nModel artifacts will be saved to:")
    print(f"{output_path}")

    return estimator


def main():
    parser = argparse.ArgumentParser(description="Submit SageMaker training job")

    # Job configuration
    parser.add_argument('--job-name', type=str, default=None,
                        help='Training job name (default: auto-generated)')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type (default: ml.g4dn.xlarge)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances (default: 1)')
    parser.add_argument('--volume-size', type=int, default=50,
                        help='EBS volume size in GB (default: 50)')
    parser.add_argument('--max-run-time', type=int, default=86400,
                        help='Max training time in seconds (default: 86400 = 24h)')
    parser.add_argument('--use-spot', action='store_true',
                        help='Use spot instances for cost savings')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--config-s3-uri', type=str, default=None,
                        help='S3 URI to augmentation config JSON file')
    parser.add_argument('--job-comment', type=str, default=None,
                        help='Optional comment about this training run (saved to S3 log)')

    args = parser.parse_args()

    estimator = submit_training_job(
        job_name=args.job_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        max_run_time=args.max_run_time,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        config_s3_uri=args.config_s3_uri,
        use_spot=args.use_spot,
        job_comment=args.job_comment,
    )


if __name__ == '__main__':
    main()
