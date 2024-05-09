FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Install software-properties-common to add repositories
RUN apt-get update && apt-get install -y software-properties-common

# Install Python to use pip
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8 python3-pip

# Set the working directory
WORKDIR /app

# Copy the model repository
COPY . .

RUN pip install -r requirements.txt

# Set the environment variable for the model repository
ENV MODEL_REPOSITORY=/app/model_repository

# Expose necessary ports
EXPOSE 8000 8001 8002

# Set the entrypoint to run Triton
ENTRYPOINT ["tritonserver", "--model-repository=/app/model_repository"]
