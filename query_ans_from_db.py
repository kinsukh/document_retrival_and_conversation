from sentence_transformers import SentenceTransformer  
from Embed_storing import index

model = SentenceTransformer('all-mpnet-base-v2')

def embed_query(query):
    query_embedding = model.encode(query)  
    return query_embedding

def search_in_pinecone(query_embedding, top_k=5, user_id=None):
    """
    Query the Pinecone index to find similar documents. Optionally filter by user_id.
    """
    query_embedding = [query_embedding.tolist()]
    filter = {}
    if user_id:
        filter = {"user_id": user_id}
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter)
    return results['matches']
