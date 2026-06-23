import os
import sys
from .rag_pipeline import RAGPipeline

def main():
    print("="*60)
    print("CHILIKA LAKE RAG SYSTEM")
    print("="*60)
    
    # Check if data folder exists
    if not os.path.exists("data"):
        print("❌ Data folder not found!")
        return
    
    # Find PDF files
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF found in 'data' folder!")
        print("📂 Please copy your PDF to the 'data' folder")
        return
    
    pdf_path = os.path.join("data", pdf_files[0])
    print(f"\n📄 Using PDF: {pdf_files[0]}")
    print(f"📊 File size: {os.path.getsize(pdf_path):,} bytes")
    
    # Initialize pipeline
    print("\n🚀 Initializing RAG Pipeline...")
    pipeline = RAGPipeline(pdf_path)
    
    # Process document
    print("\n📝 Processing document...")
    if not pipeline.process_document():
        print("❌ Failed to process document!")
        return
    
    print("\n" + "="*60)
    print("✅ READY! Ask questions about your document.")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            result = pipeline.answer_question(question, top_k=3)
            print(f"\n📝 Answer:")
            print(result['answer'])
            print("\n" + "-"*40)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()