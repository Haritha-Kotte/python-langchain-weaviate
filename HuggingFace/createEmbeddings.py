from datasets import load_dataset
import weaviate
import weaviate.classes.config as wvcc
from weaviate.auth import Auth
from langchain_huggingface import HuggingFaceEmbeddings
import time
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access environment variables
cluster_url = os.getenv('WEAVIATE_CLUSTER')
auth_key = os.getenv('WEAVIATE_KEY')

# Constants
RATE_LIMIT = 60

df = load_dataset("Shengtao/recipe")
recipes = df['train']
recipes = recipes.select(range(100))
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

texts = []
for recipe in recipes:
    data = " ".join([f"{key}: {recipe[key]}" for key in recipes.column_names])
    texts.append(data)

embedded_doc = embeddings.embed_documents(texts)
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=cluster_url,
    auth_credentials=Auth.api_key(auth_key),
    skip_init_checks=True,
)
properties = [wvcc.Property(
                name=col,
                data_type=wvcc.DataType.TEXT
            ) for col in recipes.column_names if col != "embedding"]

weaviate_client.collections.delete("RecipeHFE")
collection = weaviate_client.collections.create(
    name="RecipeHFE",
    description="A collection to store recipes",
    properties=properties + [wvcc.Property(
            name="embedding",
            data_type=wvcc.DataType.NUMBER_ARRAY
        )],
)

try:
    with weaviate_client.batch.rate_limit(requests_per_minute=RATE_LIMIT) as batch:
        for index, row in enumerate(recipes):
            data_object = {col: str(row[col]) for col in recipes.column_names}
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Create object with embedding
                    data_object['embedding'] = embedded_doc[index]

                    # Create object with embedding
                    batch.add_object(
                        properties=data_object,
                        collection="RecipeHFE",
                    )

                    print("Data uploaded successfully.")
                    break
                except weaviate.exceptions.UnexpectedStatusCodeException as e:
                    if '503' in str(e):
                        print(f"Attempt {attempt + 1}: Model is still loading, retrying...")
                        time.sleep(20)  # Wait and retry
                    if '429' in str(e):
                        # Handle rate limit error
                        retry_after = 60  # Retry-After header might be in seconds
                        print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                    else:
                        raise  # Raise if it's a different error
                
finally:
    weaviate_client.close()

