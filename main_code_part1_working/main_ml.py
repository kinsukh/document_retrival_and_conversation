import os
from loading_document_and_embedding import process
from Embed_storing import insert_embeddings_into_pinecone
from searcher_main import query_with_duckduckgo_search

# Load the document directory
document_dir = os.getenv("DOCUMENT_DIR")
bert_embeddings,documents = process(document_dir)

# making pinecone index and storing
insert_embeddings_into_pinecone(bert_embeddings, documents)
print("Embeddings stored in Pinecone")

# trying to retrive document from the pinecone databse
query = "Which parameters does the system use to predict weather?"  # change in user input


# results = query_with_duckduckgo_search(query, top_k=5)
# print(results)

def main_search_function(query,top_k=5):
    results = query_with_duckduckgo_search(query, top_k=5)
    return results


