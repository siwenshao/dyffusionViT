# Use official PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install necessary packages
RUN apt update && apt install -y git && pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/siwenshao/dyffusionViT.git /workspace/dyffusionViT

# Set working directory inside the repository
WORKDIR /workspace/dyffusionViT

# Install all dependencies
RUN pip install '.[train]'

# Keep the container running (so the job doesn't exit immediately)
CMD ["/bin/bash", "-c", "sleep infinity"]
