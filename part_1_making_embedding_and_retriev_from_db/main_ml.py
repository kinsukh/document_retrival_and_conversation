import os
from loading_document_and_embedding import process
from Embed_storing import insert_embeddings_into_pinecone
from query_ans_from_db import embed_query, search_in_pinecone

# Load the document directory
document_dir = os.getenv("DOCUMENT_DIR")
bert_embeddings,documents = process(document_dir)

# making pinecone index and storing
insert_embeddings_into_pinecone(bert_embeddings, documents)
print("Embeddings stored in Pinecone")

# trying to retrive document from the pinecone databse
query = "Which parameters does the system use to predict weather?"  # change in user input
query_embedding = embed_query(query)

results = search_in_pinecone(query_embedding, top_k=5)

# Display the results
for match in results:
    print(f"Document ID: {match['id']}, Similarity Score: {match['score']}")
    print(f"Document Content: {match['metadata']['content']}\n")