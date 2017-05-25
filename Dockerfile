FROM continuumio/miniconda3:latest

WORKDIR /root/

ENV REPO='https://github.com/takuti/docker-td-sklearn/archive/master.zip'

RUN conda install -y scikit-learn boto3 && \
    pip install td_client && \
    wget $REPO && \
    apt-get install unzip && unzip master.zip && \
    mv docker-td-sklearn-master/src .

RUN mkdir .aws/
ADD ./credentials .aws/

ENTRYPOINT ["python"]
