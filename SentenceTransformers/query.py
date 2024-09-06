from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access environment variables
cluster_url = os.getenv('WEAVIATE_CLUSTER')
auth_key = os.getenv('WEAVIATE_KEY')

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=cluster_url, 
    auth_credentials=Auth.api_key(auth_key),
    skip_init_checks=True,
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

vectorstore = WeaviateVectorStore(client=weaviate_client, index_name="RecipeST", text_key="title", embedding=embeddings)

# Perform a search query
query = "lunch with high protein potato recipe"
results = vectorstore.similarity_search(query, k=3)  # Retrieve top similar results

print("\nHere are some meal options for you:\n")
for i, result in enumerate(results):
    print(f"Option {i + 1}:")
    print("Title:", result.page_content)
    print("Ingredients:", result.metadata.get("ingredients"))
    print("Instructions:", result.metadata.get("instructions_list"))
    print("Carbohydrates:", result.metadata.get("carbohydrates_g"), "g")
    print("Fat:", result.metadata.get("fat_g"), "g")
    print("Protein:", result.metadata.get("protein_g"), "g")
    # Print other nutritional fields...
    print()

weaviate_client.close()