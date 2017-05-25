import os
import sys
import json
import pickle
import boto3
import tdclient
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def load_data(apikey, db_name, query):
    res = []
    job_id = ''

    with tdclient.Client(apikey) as td:
        job = td.query(db_name, query, type='presto')
        job.wait()
        for row in job.result():
            res.append(row)
        job_id = job.job_id

    return job_id, res


def main():
    # parse cli args
    apikey = sys.argv[1]
    db_name = sys.argv[2]
    table_name = sys.argv[3]
    params = json.loads(sys.argv[4])

    cols = ', '.join(params['features']) + ', ' + params['target']
    query = 'select %s from %s' % (cols, table_name)
    if 'limit' in params:
        query += ' limit ' + str(params['limit'])

    job_id, res = load_data(apikey, db_name, query)
    mat = np.asarray(res)
    X, y = mat[:, :-1], mat[:, -1]

    n_estimators = 10 if 'n_estimators' not in params else params['n_estimators']
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X, y)

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    boto3.setup_default_session(profile_name=os.environ['AWS_PROFILE_NAME'])
    s3 = boto3.resource('s3')
    s3.Object(os.environ['AWS_BUCKET_NAME'], job_id + '.pkl').put(Body=pickle.dumps(rf))

    print(job_id)


if __name__ == '__main__':
    main()
