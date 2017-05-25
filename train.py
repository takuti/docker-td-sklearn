import os
import sys
import json
import tdclient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


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

    dirpath = os.path.join('models', job_id)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # TODO: store the model to S3
    filepath = os.path.join(dirpath, job_id + '.pkl')
    joblib.dump(rf, filepath)


if __name__ == '__main__':
    main()
