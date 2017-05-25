```sh
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=yyyyyy
export AWS_PROFILE=default
export AWS_BUCKET=awesome-bucket
```

Copy `~/.aws/credentials` to HERE (root of this repository).

## build

```
$ docker build -t sklearn -f Dockerfile .
```

### train

```
$ docker run \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_PROFILE=${AWS_PROFILE} \
-e AWS_BUCKET=${AWS_BUCKET} \
-it sklearn src/train.py $TD_API_KEY sample_datasets nasdaq \
'{
  "features": ["open", "volume", "low", "high"],
  "target": "close",
  "limit": 100,
  "n_estimators": 16
}'
```

### predict

```
$ docker run  \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_PROFILE=${AWS_PROFILE} \
-e AWS_BUCKET=${AWS_BUCKET} \
-it sklearn src/predict.py $TD_API_KEY sample_datasets nasdaq \
'{
  "features": ["open", "volume", "low", "high"],
  "limit": 100,
  "model_name": "145080752"
}'
```