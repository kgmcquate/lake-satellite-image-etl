# FROM public.ecr.aws/emr-serverless/spark/emr-6.12.0:latest
FROM 117819748843.dkr.ecr.us-east-1.amazonaws.com/emr-serverless:latest

USER root

RUN yum install -y openssl-devel

COPY ./src/app/requirements.txt ./requirements.txt
COPY ./dist/* ./dist/

RUN /usr/local/bin/python3.11 -m pip install --upgrade pip
RUN /usr/local/bin/python3.11 -m pip install -r ./requirements.txt 
RUN /usr/local/bin/python3.11 -m pip install ./dist/*

# EMRS will run the image as hadoop
USER hadoop:hadoop