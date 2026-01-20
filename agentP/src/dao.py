from config import Config
# Unused import, remove if not needed
from pinecone import Pinecone, ServerlessSpec



def connectToVDB():
    pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)


    print("Connecting to VDB...")
    for( index_name) in pc.list_indexes():
       print(f"Found index: {index_name.name}")
    

    index = pc.Index(Config.PINECONE_INDEX_NAME)
    index.upsert(
        vectors=[
            {
                "id": "vec1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"property_type": "Apartment", "location": "Berlin", "rent": 1200}
            },
            {
                "id": "vec2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {"property_type": "House", "location": "Munich", "rent": 2500}
            },
            {
                "id": "vec3",
                "values": [0.7, 0.8, 0.9],
                "metadata": {"property_type": "Apartment", "location": "Hamburg", "rent": 1500}
            }
        ]
    )
    print("Upserted vector into VDB.")
    query_result = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=1,
        include_metadata=False
    )
    print("Query result:", query_result)
    return None     


def main():
    conn = connectToVDB()
    # use conn as needed
    print("connectToVDB() returned:", conn)


if __name__ == "__main__":
    main()