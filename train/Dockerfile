FROM continuumio/miniconda3:latest

WORKDIR /root/

ENV SCRIPT_URL='https://raw.githubusercontent.com/takuti/docker-td-sklearn/master/train/train.py'

RUN conda install -y scikit-learn && \
    pip install td_client && \
    wget $SCRIPT_URL

ENTRYPOINT ["python", "train.py"]
