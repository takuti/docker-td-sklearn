import os
import boto3
import click
import numpy as np
from sklearn.externals import joblib

from common import load_data


@click.command()
@click.option('--apikey')
@click.option('--db')
@click.option('--table')
@click.option('--feature', '-f', multiple=True)
@click.option('--limit', default=10000, type=int)
@click.option('--model', type=str)
def main(apikey, db, table, feature, limit, model):
    cols = ', '.join(feature)
    query = 'select %s from %s limit %d' % (cols, table, limit)

    _, res = load_data(apikey, db, query)
    X = np.asarray(res)

    model_filename = model + '.pkl'

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    boto3.setup_default_session(profile_name=os.environ['AWS_PROFILE'])
    s3 = boto3.resource('s3')
    with open(model_filename, 'w+b') as f:
        s3.Bucket(os.environ['AWS_BUCKET']).download_fileobj(model_filename, f)
        rf = joblib.load(f)
    os.remove(model_filename)

    rf.predict(X)
    print(rf.predict(X))


if __name__ == '__main__':
    main()
