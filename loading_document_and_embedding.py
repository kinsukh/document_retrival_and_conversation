from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Load the PDF documents from a directory
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def preprocess_text(text):
    """
    Preprocess the input text: lowercase, remove special characters, tokenize, remove stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    return documents

def generate_bert_embeddings(documents):
    """
    Generate embeddings for each document chunk using SentenceTransformer.
    """
    model = SentenceTransformer('all-mpnet-base-v2')
    # Preprocess text for each document chunk
    documents = [preprocess_text(doc.page_content) for doc in documents]
    embeddings = [model.encode(doc) for doc in documents]
    return embeddings

def process(document_dir):
    doc = read_doc(document_dir)
    documents = chunk_data(docs=doc)
    bert_embeddings = generate_bert_embeddings(documents)   
    bert_embeddings = normalize(bert_embeddings)
    return bert_embeddings, documents
