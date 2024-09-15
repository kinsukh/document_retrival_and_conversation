from sentence_transformers import SentenceTransformer  
from Embed_storing import index
model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_query(query):
    query_embedding = model.encode(query)  
    return query_embedding


# Query the Pinecone index to find similar documents
def search_in_pinecone(query_embedding, top_k=5):
    # Perform the query and retrieve top_k most similar documents
    query_embedding = [query_embedding.tolist()]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Return the list of matches
    return results['matches']


