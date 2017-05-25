import tdclient


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
