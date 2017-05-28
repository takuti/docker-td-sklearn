import os
import pickle
import time
import boto3
import tdclient
import numpy as np
from optparse import OptionParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib


def load_data(apikey, db_name, query):
    res = []

    with tdclient.Client(apikey) as td:
        job = td.query(db_name, query, type='presto')
        job.wait()
        for row in job.result():
            res.append(row)

    mat = np.asarray(res)
    X, y = mat[:, :-1], mat[:, -1]

    return X, y


def load_data_livsvm(apikey, db_name, query):
    with tdclient.Client(apikey) as td:
        job = td.query(db_name, query, type='presto')
        job.wait()
        f = open('tmp.dat', 'w')
        for row in job.result():
            if len(row) == 1:  # for target-less prediction
                feature, target = row[0], 0
            else:
                feature, target = row[0], row[1]
            if type(feature) is not list:
                continue
            print(str(target) + ' ' + ' '.join(feature), file=f)

    X, y = load_svmlight_file('tmp.dat')

    os.remove('tmp.dat')
    f.close()

    return X, y


def train(opts):
    cols = ', '.join(opts.feature) + ', ' + opts.target
    query = 'select %s from %s limit %d' % (cols, opts.table, opts.limit)

    if len(opts.feature) == 1:  # livsvm
        X, y = load_data_livsvm(opts.apikey, opts.db, query)
    else:
        X, y = load_data(opts.apikey, opts.db, query)

    rf = RandomForestRegressor(n_estimators=opts.n_estimators)
    rf.fit(X, y)

    # boto3 internally checks "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY"
    # http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
    s3 = boto3.resource('s3')
    s3.Object(os.environ['AWS_BUCKET'], opts.model + '.pkl').put(Body=pickle.dumps(rf))

    print(rf)
    print(opts.model)


def predict(opts):
    cols = ', '.join(opts.feature)
    query = 'select %s from %s limit %d' % (cols, opts.table, opts.limit)

    if len(opts.feature) == 1:  # livsvm
        X, y = load_data_livsvm(opts.apikey, opts.db, query)
    else:
        X, y = load_data(opts.apikey, opts.db, query)

    model_filename = opts.model + '.pkl'

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

    with tdclient.Client(opts.apikey) as td:
        td.import_file(opts.db, opts.table + '_predict', 'csv', 'tmp.csv')

    os.remove('tmp.csv')
    print(y)


def cli():
    parser = OptionParser()
    parser.add_option('--apikey')
    parser.add_option('--db')
    parser.add_option('--table')
    parser.add_option('--target')
    parser.add_option('--model')
    parser.add_option('-f', '--feature', action='append')
    parser.add_option('--limit', type='int', default=10000)
    parser.add_option('--n_estimators', type='int', default=10)

    opts, args = parser.parse_args()
    print(opts)

    if args[0] == 'train':
        train(opts)
    elif args[0] == 'predict':
        predict(opts)
    else:
        raise ValueError('Unsupported operation')


if __name__ == '__main__':
    cli()