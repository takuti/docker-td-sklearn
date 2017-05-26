import os
import tdclient
import numpy as np
from sklearn.datasets import load_svmlight_file


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
