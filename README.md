```
$ python train.py $TD_API_KEY sample_datasets nasdaq \
'{
  "features": ["open", "volume", "low", "high"],
  "target": "close",
	"limit": 100,
  "n_estimators": 16
}'
```