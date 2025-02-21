import os
from loading_document_and_embedding import process
from Embed_storing import insert_embeddings_into_pinecone
from searcher_main import query_with_db_search

# Load and process documents
document_dir = os.getenv("DOCUMENT_DIR")
bert_embeddings, documents = process(document_dir)

# Insert embeddings into Pinecone
insert_embeddings_into_pinecone(bert_embeddings, documents)
print("Embeddings stored in Pinecone")

def main_search_function(query):
    results = query_with_db_search(query)
    return results
