import json
import os
import boto3
import sqlmodel
import sqlalchemy

secret_arn = os.environ.get("DB_CREDS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:117819748843:secret:lake-freeze-db-creds")

db_endpoint = os.environ.get("DB_ENDPOINT" , "lake-freeze-db.cu0bcthnum69.us-east-1.rds.amazonaws.com")


print("getting creds from sm")
secret = json.loads(
        boto3.client("secretsmanager", 'us-east-1')
        .get_secret_value(SecretId=secret_arn)
        ["SecretString"]
)

db_username = secret["username"]

db_password = secret["password"]

jdbc_url = f'postgresql+psycopg2://{db_username}:{db_password}@{db_endpoint}'

connectorx_url = f"postgresql://{db_username}:{db_password}@{db_endpoint}:5432/postgres"


# print("creating engine")
engine = sqlmodel.create_engine(jdbc_url, echo=True) #/lake_freeze


