import os
import time
from pinecone import Pinecone, ServerlessSpec

index_name = "trademarkia-task2"
dimension = 768

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if index_name not in pc.list_indexes().names():
    print(f"creating Index {index_name} ...")
    pc.create_index(
        name=index_name, 
        dimension=dimension, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

print("connection made...")
def insert_embeddings_into_pinecone(embeddings, documents):
    for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
        doc_id = f"doc_{i}"
        index.upsert([(doc_id, embedding, {"content": doc.page_content })])
    print(f"Inserted {len(embeddings)} embeddings into Pinecone.")

