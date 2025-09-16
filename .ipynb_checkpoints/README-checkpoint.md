# hyperion_unet_training

This repo contains code for training Hyperion's UNet model.
This model segments vessels in ultrasound data.

Ultrasound data for training, model weights, and experiment results are not stored on GitHub.
Instead, these larger files are stored on an S3 bucket in Hyperion's AWS account called `vessel-segmentation-data`.
This bucket is organized as follows:
- `data` contains ultrasound data and labels for training, validation, and testing
- `experiments` contains TensorBoard results for training and validation runs, as well as test results on finalized model weights
- `models` contains the final model weights from experiments

The repo is organized into directories for different tasks within the training process.
See the README files within each directory for more details.
- `data_handling` contains functions for converting raw data (ultrasound imagery and labels) into a consistent format for training, as well as code for handling the train/val/test split
- `testing` contains functions for evaluating finalized models from different experiments
- `training` contains the training and validation code