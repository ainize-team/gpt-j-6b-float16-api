FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV MODEL_NAME_OR_PATH="EleutherAI/gpt-j-6B"
ENV IS_FP16="True"
ENV REVISION="float16"

# Install Python3
RUN apt-get update && apt-get -y install python3 python3-dev python3-pip
RUN alias pip=pip3
RUN alias python=python3

COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade -r ./requirements.txt

COPY ./app /app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]