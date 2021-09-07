# experimentos-mlflow

Rodando o exemplo com o dataset de vinhos rode os comandos abaixo:

```sh
docker-compose up -d
cd examples/sklearn_elasticnet_wine/
mlflow ui --backend-store-uri postgresql://postgres:postgres@localhost/mlflow --default-artifact-root s3://mlflow
```

A flag `--backend-store-uri` serve para armazenar modelos utilizando um banco postgres.
A flag `--default-artifact-root` serve para indicar a pasta onde as execuções do mlflow são armazenadas

