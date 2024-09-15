from flask import Flask, request, render_template
from part_1_making_embedding_and_retriev_from_db.query_ans_from_db import embed_query, search_in_pinecone

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form.get('text', '')
        top_k = int(request.form.get('top_k', 5))
        threshold = float(request.form.get('threshold', 0.4))
        
        if not query:
            return render_template('index.html', error="Query text is required")
        
        query_embedding = embed_query(query)
        print(f"Query Embedding: {query_embedding}")  
        
        results = search_in_pinecone(query_embedding, top_k=top_k)
        print(f"Raw Results: {results}")  
        
        filtered_results = [res for res in results if res['score'] >= threshold]
        print(f"Filtered Results: {filtered_results}")  

        if not filtered_results:
            return render_template('index.html', error="No results found")

        return render_template('index.html', results=filtered_results)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  
        return render_template('index.html', error="An error occurred: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
