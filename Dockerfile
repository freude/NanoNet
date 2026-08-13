FROM python:3.12-slim

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install pytest && \
    pip install mpi4py && \
    pip install .

# Run your script
RUN pytest
