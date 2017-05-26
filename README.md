```sh
export AWS_BUCKET=awesome-bucket
```

## build

```
$ docker build -t sklearn -f Dockerfile .
```

### train

```
$ aws ecs run-task --cluster ml-api-cluster --task-definition sklearn-train:6 --overrides \
'{
  "containerOverrides": [
    {
      "name": "sklearn",
      "command": ["--apikey","'$TD_API_KEY'","--db","takuti","--table","news20_binary","-f","features","--target","label","--limit","100","--n_estimators","16","--model","awesome_model"]
    }
  ]
}'
```

### predict

```
$ aws ecs run-task --cluster ml-api-cluster --task-definition sklearn-predict:2 --overrides \
'{
  "containerOverrides": [
    {
      "name": "sklearn",
      "command": ["--apikey","'$TD_API_KEY'","--db","takuti","--table","news20_binary","-f","features","--limit","100","--model","awesome_model"]
    }
  ]
}'
```