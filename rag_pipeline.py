from .pdf_loader import PDFLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase

class RAGPipeline:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.is_initialized = False
        
        print("\n" + "="*60)
        print("RAG PIPELINE INITIALIZATION")
        print("="*60)
        
        self.pdf_loader = PDFLoader(pdf_path)
        self.text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        
        self.chunks = []
        self.embeddings = None
        
    def process_document(self):
        print("\n" + "="*60)
        print("PROCESSING DOCUMENT")
        print("="*60)
        
        # STEP 4: Load PDF
        print("\n📄 Loading PDF...")
        text = self.pdf_loader.load_with_pdfplumber()
        if not self.pdf_loader.verify_content():
            return False
        
        # STEP 5: Split Text
        print("\n✂️ Splitting text into chunks...")
        self.chunks = self.text_splitter.split_with_metadata(text)
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        
        # STEP 6: Generate Embeddings
        print("\n🧠 Generating embeddings...")
        self.embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
        self.embedding_generator.verify_embeddings(self.embeddings)
        
        # STEP 7: Store in Vector Database
        print("\n💾 Storing in vector database...")
        self.vector_db.store_embeddings(
            embeddings=self.embeddings.tolist(),
            documents=chunk_texts
        )
        self.vector_db.create_index()
        self.vector_db.verify_database()
        
        self.is_initialized = True
        print("\n" + "="*60)
        print("✅ DOCUMENT PROCESSING COMPLETE!")
        print("="*60)
        return True
    
    def answer_question(self, question, top_k=3):
        if not self.is_initialized:
            return {"error": "Pipeline not initialized. Process document first."}
        
        print("\n🔍 Searching for answer...")
        
        # Convert question to embedding
        query_embedding = self.embedding_generator.model.encode([question])[0]
        
        # Search vector database
        results = self.vector_db.search(
            query_embedding=query_embedding.tolist(),
            n_results=top_k
        )
        
        if not results:
            answer = "I couldn't find relevant information to answer your question."
        else:
            # Combine results
            context = "\n\n".join([f"Source {i+1}: {r['text']}" for i, r in enumerate(results)])
            answer = f"Based on the document, here's what I found:\n\n{context}"
            answer += f"\n\n---\n📊 Retrieved from {len(results)} relevant sources."
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": results
        }