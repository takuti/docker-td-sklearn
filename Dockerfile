FROM continuumio/miniconda3:latest

WORKDIR /root/

RUN conda install -y scikit-learn boto3 && \
    pip install td_client

ADD . .

CMD ["/bin/sh"]
