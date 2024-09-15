from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  
model = SentenceTransformer('all-MiniLM-L6-v2') 

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    return documents

def generate_bert_embeddings(documents):
    embeddings = [model.encode(doc.page_content) for doc in documents]
    return embeddings

def process(document_dir):
    doc = read_doc(document_dir)
    documents = chunk_data(docs=doc)
    bert_embeddings = generate_bert_embeddings(documents)   
    return bert_embeddings,documents
