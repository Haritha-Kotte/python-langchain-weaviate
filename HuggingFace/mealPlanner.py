from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate.classes.init import Auth

WEAVIATE_CLUSTER = "https://q6zqx8zbrxcmb6mg4lklqw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_KEY = "tgCsONAsEGhXZam1jeGS6xUThPmxfToGT5FE"

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER,
    auth_credentials=Auth.api_key(WEAVIATE_KEY),
    skip_init_checks=True,
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

vectorstore = WeaviateVectorStore(client=weaviate_client, index_name="RecipeHFE", text_key="title", embedding=embeddings)

# Meal planner
def meal_planner():
    print("Welcome to the Meal Planner!")
    while True:
        meal_type = input("Enter the type of meal (e.g., breakfast, lunch, dinner, main dish): ")
        ingredients = input("Do you have any specific ingredients for your dish? (leave blank if you don't have any preference): ")
        nutrition_goals = input("Enter specific nutritional goals (e.g., low-carb, high-protein): ")
        any_other = input("Do you have any other preference? (leave blank if you don't have any preference): ")
        query = f"{meal_type} {ingredients} {nutrition_goals} {any_other}"
    
        results = vectorstore.similarity_search(query, k=2)
    
        print("\nHere are some meal options for you:\n")
        for i, result in enumerate(results):
            print(f"Option {i + 1}:")
            print("Title:", result.page_content)
            print("Ingredients:", result.metadata.get("ingredients"))
            print("Instructions:", result.metadata.get("instructions_list"))
            print("Calories:", result.metadata.get("calories"))
            print("Carbohydrates:", result.metadata.get("carbohydrates_g"), "g")
            print("Fat:", result.metadata.get("fat_g"), "g")
            print("Protein:", result.metadata.get("protein_g"), "g")
            # Print other nutritional fields...
            print()
            
        another = input("Do you want to plan another meal? (yes/no): ")
        if another.lower() != "yes":
            break

meal_planner()
weaviate_client.close()
