from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the .env file
load_dotenv()

# Access environment variables
cluster_url = os.getenv('WEAVIATE_CLUSTER')
auth_key = os.getenv('WEAVIATE_KEY')

# Connect to weaviate client
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=cluster_url, 
    auth_credentials=Auth.api_key(auth_key),
    skip_init_checks=True,
)

# Mention the embedding model name
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

# Initialise the vector store
vectorstore = WeaviateVectorStore(client=weaviate_client, index_name="RecipeV4", text_key="title", embedding=embeddings)
# Create the retriever to fetch relevant documents based on a query.
retriever = vectorstore.as_retriever()

# Construct a template for the RAG mode
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Show in a detailed information list format for the user to prepare the dishes and analyze the nutrition information of the dishes.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# Connect to OpenAI GPT Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
# Build the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

output = rag_chain.invoke("lunch with high protien potato recipe")
print(output)