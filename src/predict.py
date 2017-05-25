import os
import sys
import json
import boto3
import numpy as np
from sklearn.externals import joblib

from common import load_data


def main():
    # parse cli args
    apikey = sys.argv[1]
    db_name = sys.argv[2]
    table_name = sys.argv[3]
    params = json.loads(sys.argv[4])

    cols = ', '.join(params['features'])
    query = 'select %s from %s' % (cols, table_name)
    if 'limit' in params:
        query += ' limit ' + str(params['limit'])

    _, res = load_data(apikey, db_name, query)
    X = np.asarray(res)

    model_filename = params['model_name'] + '.pkl'

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    boto3.setup_default_session(profile_name=os.environ['AWS_PROFILE_NAME'])
    s3 = boto3.resource('s3')
    with open(model_filename, 'w+b') as f:
        s3.Bucket(os.environ['AWS_BUCKET_NAME']).download_fileobj(model_filename, f)
        rf = joblib.load(f)
    os.remove(model_filename)

    rf.predict(X)
    print(rf.predict(X))


if __name__ == '__main__':
    main()
