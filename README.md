```
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=yyyyyy
export AWS_PROFILE_NAME=default
export AWS_BUCKET_NAME=awesome-bucket
```

```
$ docker build -t sklearn_train -f Dockerfile .
$ docker run -it sklearn_train $TD_API_KEY sample_datasets nasdaq \
'{
  "features": ["open", "volume", "low", "high"],
  "target": "close",
  "limit": 100,
  "n_estimators": 16
}'
```