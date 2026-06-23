import os
from pypdf import PdfReader
import pdfplumber

class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text = ""
        self.pages = []
    
    def load_with_pdfplumber(self):
        """Read the PDF document and extract text from all pages"""
        print("\n=== STEP 4: LOADING PDF DOCUMENT ===")
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        self.pages.append(text)
                self.text = "\n".join(self.pages)
                print(f"✅ Loaded {len(self.pages)} pages using pdfplumber")
                return self.text
        except Exception as e:
            print(f"⚠️ Error with pdfplumber: {e}")
            return self.load_with_pypdf()
    
    def load_with_pypdf(self):
        """Fallback: Load PDF using PyPDF"""
        try:
            reader = PdfReader(self.pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.pages.append(text)
            self.text = "\n".join(self.pages)
            print(f"✅ Loaded {len(self.pages)} pages using PyPDF")
            return self.text
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return ""
    
    def verify_content(self):
        """Verify that the document content is loaded correctly"""
        print("\n=== VERIFYING DOCUMENT CONTENT ===")
        if not self.text:
            print("❌ No content loaded!")
            return False
        
        print(f"✅ Content loaded successfully!")
        print(f"📊 Total characters: {len(self.text)}")
        print(f"📊 Total words: {len(self.text.split())}")
        print(f"📊 Total pages: {len(self.pages)}")
        return True