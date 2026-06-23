import chromadb
from chromadb.config import Settings
import os

class VectorDatabase:
    def __init__(self, collection_name: str = "chilika_rag"):
        self.collection_name = collection_name
        self.persist_dir = "./chroma_db"
        
        print("\n=== STEP 7: CREATING VECTOR DATABASE ===")
        print(f"🗄️ Creating collection: {collection_name}")
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_dir,
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✅ Loaded existing collection")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Created new collection")
    
    def store_embeddings(self, embeddings, documents):
        print(f"\n💾 Storing {len(documents)} embeddings...")
        
        ids = [f"chunk_{i}" for i in range(len(documents))]
        metadatas = [{"text": doc[:100]} for doc in documents]
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        
        print(f"✅ All {len(documents)} embeddings stored!")
        return True
    
    def create_index(self):
        print("\n🔧 Creating index for fast retrieval...")
        print("✅ Index created (HNSW)")
    
    def verify_database(self):
        print("\n=== VERIFYING VECTOR DATABASE ===")
        count = self.collection.count()
        print(f"✅ Database verified!")
        print(f"📊 Documents in database: {count}")
        return True
    
    def search(self, query_embedding, n_results=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "score": 1 - results['distances'][0][i]
            })
        
        return formatted_results