import os
import pickle
import boto3
import click
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from common import load_data


@click.command()
@click.option('--apikey')
@click.option('--db')
@click.option('--table')
@click.option('--feature', '-f', multiple=True)
@click.option('--target')
@click.option('--limit', default=10000, type=int)
@click.option('--n_estimators', default=10)
def main(apikey, db, table, feature, target, limit, n_estimators):
    cols = ', '.join(feature) + ', ' + target
    query = 'select %s from %s limit %d' % (cols, table, limit)

    job_id, res = load_data(apikey, db, query)
    mat = np.asarray(res)
    X, y = mat[:, :-1], mat[:, -1]

    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X, y)

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    boto3.setup_default_session(profile_name=os.environ['AWS_PROFILE'])
    s3 = boto3.resource('s3')
    s3.Object(os.environ['AWS_BUCKET'], job_id + '.pkl').put(Body=pickle.dumps(rf))

    print(job_id)


if __name__ == '__main__':
    main()
