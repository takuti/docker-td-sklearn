```sh
export TD_API_KEY=000/aaaaa
export AWS_BUCKET=awesome-bucket
```

## build

```
$ docker build -t sklearn -f Dockerfile .
```

and push to ECR

## train

Create an ECS task definition with endpoint: `python,sklearn_cli.py,train`

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

## predict

Create an ECS task definition with endpoint: `python,sklearn_cli.py,predict`

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