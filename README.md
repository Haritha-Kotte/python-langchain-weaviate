# python-langchain-weaviate

python -m venv .venv

source .venv/bin/activate

## Install packages

pip install pipenv

pipenv install

## Setup Huggingface account for generating embeddings

- Create or login to huggingface account
- Go to settings and select Access Tokens
- Here, you’ll find your personal API key. If it's not available, you can generate a new token by clicking New token, then name it and select the appropriate scope.
- Copy the API key and store it in .env file with the name HUGGINGFACE_NEW_APIKEY.

## Setup Weaviate for storing embeddings

- Create or login to weaviate account.
- Once logged in, click on New Cluster to create a Weaviate instance.
- Follow the setup steps to configure your Weaviate cluster. You can choose between different configurations (sandbox, enterprise, etc.).
- After setting up your instance, go to the Cluster Details page of the instance you created.
- In the API Keys section, you’ll find your API key.
- If you haven't created an API key yet, you can generate a new one by clicking on the Generate API Key button.
- Copy the API key and store it in .env file with the name WEAVIATE_KEY.
- Copy the REST Endpoint url and store it in .env file with the name WEAVIATE_CLUSTER.

## Setup Open AI account

use the API key in .env file with the name OPENAI_API_KEY if you want to execute the RAG pipeline

## To run a file

cd path/to/your/folder && python filename.py
