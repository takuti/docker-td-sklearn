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
-it sklearn src/train.py \
--apikey $TD_API_KEY --db takuti --table news20_binary \
-f features \
--target label \
--limit 100 \
--n_estimators 16
```

### predict

```
$ docker run  \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_PROFILE=${AWS_PROFILE} \
-e AWS_BUCKET=${AWS_BUCKET} \
-it sklearn src/predict.py \
--apikey $TD_API_KEY --db takuti --table news20_binary \
-f features \
--limit 100 \
--model 145250382
```