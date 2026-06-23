import subprocess
import sys
import os

def main():
    print("="*60)
    print("RAG PROJECT - PDF QUESTION ANSWERING")
    print("="*60)
    
    # Check if any PDF exists in data folder
    data_folder = "data"
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("\n❌ No PDF found in 'data' folder!")
        print("📂 Please copy your PDF to the 'data' folder")
        return
    
    pdf_path = os.path.join(data_folder, pdf_files[0])
    print(f"\n✅ Found PDF: {pdf_files[0]}")
    print(f"📄 File size: {os.path.getsize(pdf_path):,} bytes")
    
    print("\n📦 Checking dependencies...")
    try:
        import sentence_transformers
        print("✅ Dependencies found")
    except ImportError:
        print("❌ Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    print("\n🚀 Starting RAG Pipeline...")
    subprocess.run([sys.executable, "-m", "src.main"], check=True)

if __name__ == "__main__":
    main()