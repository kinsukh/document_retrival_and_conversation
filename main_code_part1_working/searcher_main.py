import pandas as pd
from query_ans_from_db import embed_query, search_in_pinecone
from duckduckgo_search import DDGS

def query_with_duckduckgo_search(query, top_k=5):
    """
    This function first queries the Pinecone database for the given query. 
    Regardless of the Pinecone results, it also performs a search using DuckDuckGo.
    
    Args:
    query (str): The user's input query.
    top_k (int): The number of top results to retrieve from Pinecone.

    Returns:
    str: Combined results from Pinecone and DuckDuckGo search.
    """
    # Get the embedding for the query
    query_embedding = embed_query(query)

    # Search in Pinecone
    results = search_in_pinecone(query_embedding, top_k=top_k)

    # If Pinecone results are found, display them
    if results:
        result_str = "Results from Pinecone database:\n"
        for match in results:
            result_str += f"Document ID: {match['id']}, Similarity Score: {match['score']}\n"
            result_str += f"Document Content: {match['metadata']['content']}\n\n"
    else:
        result_str = "No results found in Pinecone database.\n"

    # Perform DuckDuckGo search regardless of Pinecone results
    result_str += "\nSearching DuckDuckGo...\n"
    duckduckgo_results = search_query(query)
    
    if not duckduckgo_results.empty:
        result_str += f"Top DuckDuckGo Results:\n"
        for i in range(len(duckduckgo_results)):
            result_str += f"Title: {duckduckgo_results.iloc[i]['title']}\n"
            result_str += f"Snippet: {duckduckgo_results.iloc[i]['body']}\n"
            result_str += f"Link: {duckduckgo_results.iloc[i]['href']}\n\n"

    else:
        result_str += "No results found on DuckDuckGo."

    return result_str

def search_query(query):
    """
    Searches the query using DuckDuckGo and returns the top result.
    
    Args:
    query (str): The user's input query.
    
    Returns:
    pd.DataFrame: A DataFrame containing the top search result from DuckDuckGo.
    """
    results = DDGS().text(
        keywords=str(query),
        max_results=3,
        region='wt-wt',  # Worldwide search
        timelimit='7d',  # Searches within the last 7 days
        safesearch='on'  # SafeSearch on
    )
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df
