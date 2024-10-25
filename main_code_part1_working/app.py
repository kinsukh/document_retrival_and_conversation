from flask import Flask, request, render_template
from main_ml import main_search_function
import logging
import re
from flask_caching import Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure caching (using SimpleCache, but you can use Redis or Memcached for production)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
cache.init_app(app)

def parse_search_results(raw_results):
    """Parse and structure the search results"""
    try:
        # Split the results string into Pinecone and DuckDuckGo sections
        sections = raw_results.split("\nSearching DuckDuckGo...")
        
        pinecone_results = sections[0].strip()
        ddg_results_raw = sections[1].strip() if len(sections) > 1 else ""
        
        structured_results = {
            "pinecone": [],
            "duckduckgo": []
        }

        # Parse Pinecone results
        if "Results from Pinecone database:" in pinecone_results:
            results = pinecone_results.split("Document ID:")[1:]  # Skip the header
            for result in results:
                parts = result.split("Document Content:", 1)
                if len(parts) == 2:
                    id_score = parts[0].strip()
                    content = parts[1].strip()
                    
                    # Extract score from id_score
                    score = float(id_score.split("Similarity Score:", 1)[1].strip())
                    
                    structured_results["pinecone"].append({
                        "content": content,
                        "score": score
                    })

        # Parse multiple DuckDuckGo results
        if ddg_results_raw:
            # Split the DuckDuckGo results into individual results
            ddg_entries = re.split(r'(?=Title:)', ddg_results_raw)
            
            for entry in ddg_entries:
                if 'Title:' in entry:
                    try:
                        title = re.search(r'Title:(.*?)(?=Snippet:|$)', entry, re.DOTALL)
                        snippet = re.search(r'Snippet:(.*?)(?=Link:|$)', entry, re.DOTALL)
                        link = re.search(r'Link:(.*?)(?=Title:|$)', entry, re.DOTALL)
                        
                        if title and snippet:
                            structured_results["duckduckgo"].append({
                                "title": title.group(1).strip(),
                                "snippet": snippet.group(1).strip(),
                                "link": link.group(1).strip() if link else ""
                            })
                    except Exception as e:
                        logger.error(f"Error parsing DuckDuckGo result entry: {e}")
                        continue

        return structured_results
    except Exception as e:
        logger.error(f"Error parsing results: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get search parameters
        query = request.form.get('text', '').strip()
        top_k = int(request.form.get('top_k', 5))
        threshold = float(request.form.get('threshold', 0.4))
        
        if not query:
            return render_template('index.html',
                                   error="Please enter a search query",
                                   last_query=query,
                                   last_top_k=top_k,
                                   last_threshold=threshold)

        logger.info(f"Processing search query: '{query}' with top_k={top_k}, threshold={threshold}")
        
        # Check if results are cached
        cache_key = f"{query}_{top_k}_{threshold}"
        cached_results = cache.get(cache_key)
        if cached_results:
            logger.info(f"Returning cached results for query: '{query}'")
            return render_template('index.html',
                                   results=cached_results,
                                   last_query=query,
                                   last_top_k=top_k,
                                   last_threshold=threshold)
        
        # Get search results
        raw_results = main_search_function(query, top_k=top_k)
        logger.debug(f"Raw results received: {raw_results}")
        
        # Parse and structure the results
        structured_results = parse_search_results(raw_results)
        
        if not structured_results:
            return render_template('index.html',
                                   error="Error processing results",
                                   last_query=query,
                                   last_top_k=top_k,
                                   last_threshold=threshold)

        # Filter Pinecone results by threshold
        if structured_results["pinecone"]:
            structured_results["pinecone"] = [
                r for r in structured_results["pinecone"]
                if r["score"] >= threshold
            ]

        logger.info(f"Returning {len(structured_results['pinecone'])} Pinecone results and "
                    f"{len(structured_results['duckduckgo'])} DuckDuckGo results")
        
        # Cache the results
        cache.set(cache_key, structured_results)

        return render_template('index.html',
                               results=structured_results,
                               last_query=query,
                               last_top_k=top_k,
                               last_threshold=threshold)
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return render_template('index.html',
                               error=f"An error occurred: {str(e)}",
                               last_query=query if 'query' in locals() else '',
                               last_top_k=top_k if 'top_k' in locals() else 5,
                               last_threshold=threshold if 'threshold' in locals() else 0.4)

if __name__ == '__main__':
    app.run(debug=True)
