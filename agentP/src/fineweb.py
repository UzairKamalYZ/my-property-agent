import pinecone
from config import Config
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer


def huggingfaceFineWebdataset():
    # Example implementation (replace with actual logic)
    print("Calling huggingfaceFineWebdataset")

    fw = load_dataset("HuggingFaceFw/fineweb", split="train", streaming=True, name="sample-10BT")
    print(fw.features)

    subset_size = 1000
    vectors_to_upsert = []
    for i, item in enumerate(fw):
        print(item['text'])
        if i >= subset_size:  # Just print first 3 items for demonstration
            break
        text = item['text']
        unique_id = str(item['id'])
        language = item['language']
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode(text,show_progress_bar=False).tolist() 
        
        metadata = {"language": language}
        vectors_to_upsert.append((unique_id, embedding, metadata))
        batch_size = 1000
        index = getIndexForUpsertRecords(model)

        for i in range(0,len(vectors_to_upsert),batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            index.upsert(vectors=batch)
        print("Upserted vectors into Pinecone index.")

def getIndexForUpsertRecords(model):
    pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
    print("Connecting to Pinecone...")
    if "text" not in pc.list_indexes().names():
        pc.create_index(
        name="text",
        dimension=model.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(
        cloud='aws', region='us-east-1')
    )
    index = pc.Index("text")
    return index
if __name__ == "__main__":
    huggingfaceFineWebdataset()