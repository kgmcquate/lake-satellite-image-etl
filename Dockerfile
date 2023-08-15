FROM public.ecr.aws/emr-serverless/spark/emr-6.10.0:latest

USER root

COPY ./src/app/requirements.txt ./requirements.txt

COPY ./dist/* ./dist/

# install python 3
# RUN yum install -y gcc openssl-devel bzip2-devel libffi-devel tar gzip wget make
# RUN wget https://www.python.org/ftp/python/3.9.0/Python-3.10.0.tgz && \
#     tar xzf Python-3.10.0.tgz && cd Python-3.9.0 && \
#     ./configure --enable-optimizations && \
#     make altinstall

RUN pip3 install -r requirements.txt && pip3 install ./dist/*

# EMRS will run the image as hadoop
USER hadoop:hadoop