import os
import pickle
import boto3
from optparse import OptionParser
from sklearn.ensemble import RandomForestRegressor

from common import load_data, load_data_livsvm


def main():
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

    print(opts.model)


if __name__ == '__main__':
    main()
