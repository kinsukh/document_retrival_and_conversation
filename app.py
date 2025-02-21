import os
import time
import logging
from flask import Flask, request, render_template, jsonify, abort
from flask_caching import Cache
from background_scraper import start_background_scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
cache.init_app(app)

# Global dictionary for tracking user request counts
user_request_count = {}
MAX_REQUESTS_PER_USER = 5

# Start background scraper thread
start_background_scraper()

def parse_search_results(raw_results):
    """Parse and structure the search results (Pinecone only)"""
    try:
        structured_results = {"pinecone": []}
        if "Results from Pinecone database:" in raw_results:
            results = raw_results.split("Document ID:")[1:]
            for result in results:
                parts = result.split("Document Content:", 1)
                if len(parts) == 2:
                    id_score = parts[0].strip()
                    content = parts[1].strip()
                    score = float(id_score.split("Similarity Score:", 1)[1].strip())
                    structured_results["pinecone"].append({"content": content, "score": score})
        return structured_results
    except Exception as e:
        logger.error(f"Error parsing results: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Health endpoint returning a random status message."""
    import random
    responses = ["Healthy", "Alive", "Running", "All systems operational"]
    return jsonify({"status": random.choice(responses)}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    try:
        user_id = request.form.get('user_id', '').strip()
        if not user_id:
            return render_template('index.html', error="User ID is required.")

        query = request.form.get('text', '').strip()
        threshold = float(request.form.get('threshold', 0.4))

        # Process the file if one was uploaded
        file = request.files.get('file')
        file_uploaded = False
        if file and file.filename != "":
            from document_handler import DocumentHandler
            upload_folder = os.path.join(os.getcwd(), "uploads")
            doc_handler = DocumentHandler(upload_folder=upload_folder)

            success, result = doc_handler.save_file(file)
            if not success:
                return render_template('index.html', error=result, user_id=user_id)
            filepath = result

            success, message, text = doc_handler.process_document(filepath)
            if not success:
                return render_template('index.html', error=message, user_id=user_id)

            # Clean up the file after processing
            doc_handler.cleanup_file(filepath)

            # Define a simple Document class with a metadata attribute
            class Document:
                def __init__(self, page_content, metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata if metadata is not None else {}

            doc_obj = Document(page_content=text)

            # Process the document: chunk it and generate embeddings
            from loading_document_and_embedding import chunk_data, generate_bert_embeddings
            documents = chunk_data([doc_obj])
            embeddings = generate_bert_embeddings(documents)
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings)

            # Insert embeddings into Pinecone with the associated user_id
            from Embed_storing import insert_embeddings_into_pinecone
            insert_embeddings_into_pinecone(embeddings, documents, user_id=user_id)
            file_uploaded = True

        # If a query is provided, perform the search
        if query:
            # Rate limiting logic
            count = user_request_count.get(user_id, 0)
            if count >= MAX_REQUESTS_PER_USER:
                logger.warning(f"User {user_id} exceeded rate limit.")
                abort(429, description="Rate limit exceeded. Maximum 5 requests allowed per user.")
            user_request_count[user_id] = count + 1

            from searcher_main import query_with_db_search
            raw_results = query_with_db_search(query, user_id=user_id)
            structured_results = parse_search_results(raw_results)
            if structured_results and "pinecone" in structured_results:
                structured_results["pinecone"] = [r for r in structured_results["pinecone"] if r["score"] >= threshold]
            else:
                structured_results = {"pinecone": []}

            # Generate a conversational response using the database results
            from chat_response import get_chat_response
            chat_response_text = get_chat_response(query, structured_results)
            elapsed_time = time.time() - start_time
            logger.info(f"Inference time: {elapsed_time:.2f} seconds")

            message_text = "Document uploaded and processed successfully. " if file_uploaded else ""
            return render_template('index.html', 
                                   results=structured_results, 
                                   chat_response=chat_response_text,
                                   last_query=query, 
                                   last_threshold=threshold, 
                                   user_id=user_id, 
                                   message=message_text)
        else:
            # No query provided. If a file was processed, return a success message.
            if file_uploaded:
                return render_template('index.html', message="Document uploaded and processed successfully.", user_id=user_id)
            else:
                return render_template('index.html', error="Please enter a search query or upload a document.", user_id=user_id)
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return render_template('index.html', 
                               error=f"An error occurred: {str(e)}",
                               last_query=request.form.get('text', ''),
                               last_threshold=request.form.get('threshold', 0.4),
                               user_id=request.form.get('user_id', ''))

if __name__ == '__main__':
    app.run(debug=True)
