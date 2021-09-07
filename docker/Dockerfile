FROM python:3.8

EXPOSE 5000

RUN mkdir /usr/src/app

COPY . /usr/src/app

WORKDIR /usr/src/app

RUN pip install -r requirements.txt

ENTRYPOINT ["mlflow", "ui", "--backend-store-uri", \
            "postgresql://postgres:postgres@db/mlflow", \
            "--default-artifact-root", "s3://mlflow", \
            "--host", "0.0.0.0" \
]