# SageMaker PyTorch Training Container
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all training code
COPY training/ ./training/
COPY data_handling/ ./data_handling/
COPY configs/ ./configs/

# Set Python path so imports work correctly
ENV PYTHONPATH=/opt/ml/code:/opt/ml/code/training

# SageMaker expects the training script to be at this path
ENV SAGEMAKER_PROGRAM=training/train.py

# Default command (SageMaker will override this)
ENTRYPOINT ["python", "training/train.py"]
