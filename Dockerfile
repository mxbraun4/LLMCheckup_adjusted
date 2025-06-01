# Docker file adapted from this tutorial https://github.com/bennzhang/docker-demo-with-simple-python-app
FROM python:3.10.13

# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt . 
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install mysqlclient

#COPY utils/dependency.sh /usr/src/app/

# upgrade pip (you already have this) then

# 1) Install git
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/bigscience-workshop/petals.git /tmp/petals \
 && pip install /tmp/petals \
 && rm -rf /tmp/petals


RUN python -m nltk.downloader omw-1.4 punkt \
 && pip install -U spacy \
 && python -m spacy download en_core_web_sm

#RUN bash dependency.sh
#RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-mpnet-base-v2")'
# Copying src code to Container

COPY . .

# Application Environment variables
#ENV APP_ENV development
ENV PORT 4000

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
#CMD gunicorn -b :$PORT -c gunicorn.conf.py main:app
CMD python -m gunicorn --timeout 0 -b :$PORT flask_app:app

