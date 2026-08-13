FROM python:3.12-slim

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*


# Create Python virtual environment
RUN python -m venv /opt/venv

# Use the virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/freude/NanoNet.git && \
    cd NanoNet && ls

WORKDIR /app/NanoNet

# Install dependencies
RUN pip install --upgrade pip && \
    pip install pytest && \
    pip install mpi4py && \
    pip install .

# Run your script
RUN pytest
