from sentence_transformers import SentenceTransformer  
from Embed_storing import index
model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_query(query):
    query_embedding = model.encode(query)  
    return query_embedding


def search_in_pinecone(query_embedding, top_k=5):
    query_embedding = [query_embedding.tolist()]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']


