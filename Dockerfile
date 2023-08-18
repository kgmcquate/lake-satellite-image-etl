# FROM public.ecr.aws/emr-serverless/spark/emr-6.12.0:latest
FROM 117819748843.dkr.ecr.us-east-1.amazonaws.com/emr-serverless:latest

USER root



COPY ./src/app/requirements.txt ./requirements.txt
COPY ./dist/* ./dist/

# RUN yum install -y openssl-devel
RUN python3.11 -V 
# RUN python3 -m pip install --upgrade pip
# RUN which python3
RUN python3.11 -m pip install -r ./requirements.txt 
RUN python3.11 -m pip install ./dist/*

# EMRS will run the image as hadoop
USER hadoop:hadoop