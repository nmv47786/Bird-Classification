FROM nvcr.io/nvidia/tritonserver:24.03-py3

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8 python3-pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENV MODEL_REPOSITORY=/app/model_repository

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver", "--model-repository=/app/model_repository"]
