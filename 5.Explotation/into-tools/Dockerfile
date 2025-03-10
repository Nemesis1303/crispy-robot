
# Use official ubuntu image with CUDA support
#FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/torch/

# Install Vim, Git, Java and Ant
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim git openjdk-8-jdk ant

# Install build dependencies for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libbz2-dev \
    libenchant-dev \
    libffi-dev \
    libgdbm-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    tzdata \
    zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10.11
RUN wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
    tar xzf Python-3.10.11.tgz && \
    cd Python-3.10.11 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip && \
    cd .. && \
    rm -rf Python-3.10.11.tgz Python-3.10.11

# Install other necessary dependencies
RUN apt-get update && apt-get install hunspell-es

WORKDIR /into-tools

COPY . .

RUN mkdir -p /data/source

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[complete]"
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_md

# Clone the Mallet repository and build it
RUN mkdir -p /app && \
    git clone https://github.com/mimno/Mallet.git /app/Mallet && \
    cd /app/Mallet && \
    ant

# Change back to the working directory
WORKDIR /into-tools

# Download and cache the sentence transformer model
ARG MODEL_NAME=paraphrase-distilroberta-base-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"
RUN python initialize_st.py

EXPOSE 102

CMD ["python", "app.py"]
