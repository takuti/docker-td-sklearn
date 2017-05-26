import os
import pickle
import boto3
import click
from sklearn.ensemble import RandomForestRegressor

from common import load_data, load_data_livsvm


@click.command()
@click.option('--apikey')
@click.option('--db')
@click.option('--table')
@click.option('--feature', '-f', multiple=True)
@click.option('--target')
@click.option('--limit', default=10000, type=int)
@click.option('--n_estimators', default=10)
@click.option('--model', type=str)
def main(apikey, db, table, feature, target, limit, n_estimators, model):
    cols = ', '.join(feature) + ', ' + target
    query = 'select %s from %s limit %d' % (cols, table, limit)

    if len(feature) == 1:  # livsvm
        X, y = load_data_livsvm(apikey, db, query)
    else:
        X, y = load_data(apikey, db, query)

    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X, y)

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    s3 = boto3.resource('s3')
    s3.Object(os.environ['AWS_BUCKET'], model + '.pkl').put(Body=pickle.dumps(rf))

    print(model)


if __name__ == '__main__':
    main()
