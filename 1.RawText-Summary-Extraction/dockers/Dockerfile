FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update
RUN apt-get install -y build-essential libpoppler-cpp-dev pkg-config poppler-utils 
RUN pip3 install -r /app/requirements.txt

