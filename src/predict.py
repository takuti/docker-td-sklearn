import os
import time
import boto3
import click
import tdclient
from sklearn.externals import joblib

from common import load_data, load_data_livsvm


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

    if len(feature) == 1:  # livsvm
        X, y = load_data_livsvm(apikey, db, query)
    else:
        X, y = load_data(apikey, db, query)

    model_filename = model + '.pkl'

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    s3 = boto3.resource('s3')
    with open(model_filename, 'w+b') as f:
        s3.Bucket(os.environ['AWS_BUCKET']).download_fileobj(model_filename, f)
        rf = joblib.load(f)
    os.remove(model_filename)

    y = rf.predict(X)

    with open('tmp.csv', 'w') as f:
        f.write('time,predict\n')
        t = int(time.time())
        for yi in y:
            f.write('%d,%f\n' % (t, yi))

    with tdclient.Client(apikey) as td:
        td.import_file(db, table + '_predict', 'csv', 'tmp.csv')

    os.remove('tmp.csv')
    print(y)


if __name__ == '__main__':
    main()
