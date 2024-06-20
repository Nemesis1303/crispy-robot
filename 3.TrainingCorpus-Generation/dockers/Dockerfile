FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update
RUN pip3 install -r /app/requirements.txt
