import os
import time
import boto3
import tdclient
from optparse import OptionParser
from sklearn.externals import joblib

from common import load_data, load_data_livsvm


def main():
    parser = OptionParser()
    parser.add_option('--apikey')
    parser.add_option('--db')
    parser.add_option('--table')
    parser.add_option('--model')
    parser.add_option('-f', '--feature', action='append')
    parser.add_option('--limit', type='int', default=10000)

    opts, args = parser.parse_args()

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


if __name__ == '__main__':
    main()
