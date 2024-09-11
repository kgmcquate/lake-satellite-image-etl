FROM public.ecr.aws/emr-serverless/spark/emr-6.10.0:latest

USER root

COPY ./src/app/requirements.txt ./requirements.txt

COPY ./dist/* ./dist/

RUN pip3 install -r requirements.txt && pip3 install ./dist/*

USER hadoop:hadoop