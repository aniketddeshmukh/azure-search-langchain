# -*- coding: utf-8 -*-

import os
from azure.storage.blob import BlobServiceClient
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Set up Azure Blob Storage connection string as an environment variable
os.environ['AZURE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=poctrialstorage;AccountKey=ACCOUNT_KEY;EndpointSuffix=core.windows.net'
connection_string = os.getenv('AZURE_CONNECTION_STRING')

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "pocdocs"
blob_name = "sebi_mutual_fund.pdf"

# Download the PDF from Azure Blob Storage
def download_pdf():
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open("downloaded_pdf.pdf", "wb") as pdf_file:
        pdf_file.write(blob_client.download_blob().readall())

# Initialize the OpenAI client for embeddings
def initialize_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint="https://openaiinstancetrial.openai.azure.com/",
        api_key="YOUR_API_KEY",
        openai_api_version="2023-05-15",
        azure_deployment='poc-text-embedding'
    )

# Initialize the search client
def initialize_search_client():
    return SearchClient(
        endpoint="https://trialai.search.windows.net/",
        index_name="vector-1724859532618",
        credential=AzureKeyCredential("SEARCH_API_KEY")
    )

# Perform a search to get relevant document IDs
def search_index(search_client, query_text):
    search_results = search_client.search(
        search_text=query_text,
        select=["chunk_id", "title", "parent_id", "chunk", "metadata_storage_path", "text_vector"]
    )

    results = []
    for result in search_results:
        results.append(result)

    return results

# Generate a response using OpenAI model
def generate_response(client, prompt):
    response = client.completions.create(
        model="poc-gpt-35",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Handle user query and generate response based on retrieved documents
def handle_user_query(search_client, openai_client, query_text):
    search_results = search_index(search_client, query_text)
    context = "\n".join([result.get('chunk', '') for result in search_results])
    prompt = f"Based on the following information, answer the user's query:\n\n{context}\n\nQuery: {query_text}\nAnswer:"
    response = generate_response(openai_client, prompt)

    return response

if __name__ == "__main__":
    download_pdf()  # Download PDF file first

    # Initialize clients
    embeddings_client = initialize_embeddings()
    search_client = initialize_search_client()

    # Example user query
    user_query = "What should an investor look into an offer document?"

    # Get response for user query
    response = handle_user_query(search_client, embeddings_client, user_query)

    print(response)
