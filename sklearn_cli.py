import os
import pickle
import time
import boto3
import tdclient
import numpy as np
from optparse import OptionParser, OptionGroup
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib


def load_data(apikey, db_name, query, is_libsvm=False):
    td = tdclient.Client(apikey=apikey)

    job = td.query(db_name, query, type='presto')
    job.wait()

    if is_libsvm:
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
    else:
        res = []
        for row in job.result():
            res.append(row)
        mat = np.asarray(res)
        X, y = mat[:, :-1], mat[:, -1]

    return X, y


def train(opts):
    cols = ', '.join(opts.feature) + ', ' + opts.target
    query = 'select %s from %s limit %d' % (cols, opts.table, opts.limit)

    is_libsvm = len(opts.feature) == 1
    X, y = load_data(opts.apikey, opts.db, query, is_libsvm)

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

    is_libsvm = len(opts.feature) == 1
    X, y = load_data(opts.apikey, opts.db, query, is_libsvm)

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
    parser = OptionParser(usage='usage: %prog [options] train/predict')
    parser.add_option('--apikey',
                      help='Treasure Data API key')
    parser.add_option('--db',
                      help='Source database name')
    parser.add_option('--table',
                      help='Source table name (and destination for prediction)')
    parser.add_option('-f', '--feature', action='append',
                      help='Column names used as features')
    parser.add_option('--limit', type='int', default=10000,
                      help='Number of rows used for training/prediction')
    parser.add_option('--model',
                      help='Model name stored as a .pkl file')

    group = OptionGroup(parser, 'Options for training')
    group.add_option('--target',
                     help='Column name used as a response variable')
    group.add_option('--n_estimators', type='int', default=10,
                     help='Number of estimators for RandomForestRegressor')
    parser.add_option_group(group)

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
