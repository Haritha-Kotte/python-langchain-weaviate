from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import Auth
import weaviate.classes.config as wvcc
from sklearn.preprocessing import normalize
import time
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access environment variables
cluster_url = os.getenv('WEAVIATE_CLUSTER')
auth_key = os.getenv('WEAVIATE_KEY')

df = load_dataset("Shengtao/recipe")
recipes = df['train']
recipes = recipes.select(range(100))

model = SentenceTransformer('all-mpnet-base-v2')

def generate_embeddings(examples):
    combined_text = [" ".join([f"{col}: {examples[col][i]}" for col in recipes.column_names if col != 'embeddings']) for i in range(len(examples[recipes.column_names[0]]))]

    embeddings = model.encode(combined_text, show_progress_bar=True)
    embeddings = normalize(embeddings)
    examples["embedding"] = [embedding.tolist() for embedding in embeddings]
    
    return examples

recipes = recipes.map(generate_embeddings, batched=True)

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=cluster_url,
    auth_credentials=Auth.api_key(auth_key),
    skip_init_checks=True,
)

properties = [wvcc.Property(
                name=col,
                data_type=wvcc.DataType.TEXT
            ) for col in recipes.column_names if col != "embedding"]

weaviate_client.collections.delete("RecipeST")

collection = weaviate_client.collections.create(
    name="RecipeST",
    description="A collection to store recipes",
    properties=properties + [wvcc.Property(
            name="embedding",
            data_type=wvcc.DataType.NUMBER_ARRAY
        )],
    vector_index_config=wvcc.Configure.VectorIndex.hnsw(
        distance_metric=wvcc.VectorDistances.COSINE
    ),
)

try:
    with weaviate_client.batch.dynamic() as batch:
        for index, row in enumerate(recipes):
            data_object = {col: str(row[col]) for col in recipes.column_names if col != "embedding"}
            data_object['embedding'] = row['embedding']
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    batch.add_object(
                        properties=data_object,
                        collection="RecipeST",
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
