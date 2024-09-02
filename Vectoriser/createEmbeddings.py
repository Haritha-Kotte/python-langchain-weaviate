import weaviate
import weaviate.classes.config as wvcc
from weaviate.auth import Auth

from datasets import load_dataset
import time

# Constants
RATE_LIMIT = 10  # Number of rows per chunk
WEAVIATE_CLUSTER = "https://q6zqx8zbrxcmb6mg4lklqw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = "tgCsONAsEGhXZam1jeGS6xUThPmxfToGT5FE"
HUGGINGFACE_NEW_APIKEY = "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"

df = load_dataset("Shengtao/recipe")
recipes = df['train']
recipes = recipes.select(range(100))

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER,
    auth_credentials=Auth.api_key(WEAVIATE_KEY),
    headers={"X-HuggingFace-Api-Key": HUGGINGFACE_NEW_APIKEY},
    skip_init_checks=True,
)

properties = [wvcc.Property(
                name=col,
                data_type=wvcc.DataType.TEXT
            ) for col in recipes.column_names if col != "embedding"]

weaviate_client.collections.delete("RecipeV4")
# Note that you can use `client.collections.create_from_dict()` to create a collection from a v3-client-style JSON object
collection = weaviate_client.collections.create(
    name="RecipeV4",
    description="A collection to store recipes",
    vectorizer_config=wvcc.Configure.Vectorizer.text2vec_huggingface(
        model="sentence-transformers/all-mpnet-base-v2",
        vectorize_collection_name=True
    ),
    properties=properties
)

try:
    with weaviate_client.batch.rate_limit(requests_per_minute=RATE_LIMIT) as batch:  # or <collection>.batch.rate_limit()

        for index, row in enumerate(recipes):
            data_object = {col: str(row[col]) for col in recipes.column_names}
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    batch.add_object(properties=data_object, collection="RecipeV4")
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

