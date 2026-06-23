from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        self.model_name = model_name
        print("\n=== STEP 6: GENERATING EMBEDDINGS ===")
        print(f"🔧 Loading model: {model_name}")
        print("⏳ Downloading... (1-2 minutes first time)")
        
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded!")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts):
        print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"✅ Embeddings generated!")
        print(f"📊 Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def verify_embeddings(self, embeddings):
        print("\n=== VERIFYING EMBEDDINGS ===")
        if embeddings is None or len(embeddings) == 0:
            print("❌ No embeddings generated!")
            return False
        print(f"✅ Embeddings verified!")
        print(f"📊 Number of embeddings: {len(embeddings)}")
        print(f"📊 Embedding dimension: {embeddings.shape[1]}")
        return True