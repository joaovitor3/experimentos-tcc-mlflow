# experimentos-mlflow

Para se executar o projeto deve se executar o seguinte comando:

```sh
docker-compose up -d
``` 

Após isso, acesse o seu navegador em `http://localhost:9000` para acessar o `MinIo` que é utilizado para armazenar os artefatos dos modelos do MLFlow com as credenciais localizadas no `docker-compose.yml`:

```sh
minio
minio123
```

Com o login realizado, acesse `http://localhost:9001/buckets` e clique em **Create Bucket** para realizar a criação de um Bucket. Crie o bucket com o nome `mlflow` para utilizar as configurações padrões do projeto. 

Após isso, acesse `http://localhost:9001/users` e clique em **Create User** para criar um usuário. Selecione todas as policies disponíveis e utilize as credenciais `mlflow-integration-access-key` e `mlflow-integration-secret-key` para utilizar as configurações padrões do projeto.

Com todos os passos realizados execute os seguintes comandos para treinar um modelo.

``` 
docker-compose exec mlflow-ui bash
python examples/sklearn_elasticnet_wine/train.py
```

Também é possível executar os seguintes comandos para realizar a seleção do melhor modelo para enviá-lo para o ambiente de produção.

``` 
docker-compose exec mlflow-ui bash
python examples/sklearn_elasticnet_wine/select_model.py
```
