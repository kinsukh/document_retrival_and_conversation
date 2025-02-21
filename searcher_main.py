from query_ans_from_db import embed_query, search_in_pinecone

def query_with_db_search(query, user_id=None):
    """
    Query the Pinecone database with a default top_k of 5.
    If a user_id is provided, filter the results accordingly.
    Returns a string of results from the Pinecone database.
    """
    top_k = 5
    query_embedding = embed_query(query)
    results = search_in_pinecone(query_embedding, top_k=top_k, user_id=user_id)
    if results:
        result_str = "Results from Pinecone database:\n"
        for match in results:
            result_str += f"Document ID: {match['id']}, Similarity Score: {match['score']}\n"
            result_str += f"Document Content: {match['metadata']['content']}\n\n"
    else:
        result_str = "No results found in Pinecone database.\n"
    return result_str
