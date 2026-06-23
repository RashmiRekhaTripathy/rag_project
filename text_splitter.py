from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_text(self, text: str):
        print("\n=== STEP 5: SPLITTING TEXT INTO CHUNKS ===")
        print(f"📝 Original text length: {len(text)} characters")
        
        chunks = self.splitter.split_text(text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        print(f"✅ Text split into {len(chunks)} chunks")
        print(f"📊 Chunk overlap: {self.chunk_overlap} characters")
        
        chunk_lengths = [len(c) for c in chunks]
        print(f"📊 Average chunk size: {sum(chunk_lengths)/len(chunks):.0f} characters")
        print(f"📊 Min chunk size: {min(chunk_lengths)} characters")
        print(f"📊 Max chunk size: {max(chunk_lengths)} characters")
        
        return chunks
    
    def split_with_metadata(self, text: str):
        chunks = self.split_text(text)
        chunks_with_meta = []
        for i, chunk in enumerate(chunks):
            chunks_with_meta.append({
                "chunk_id": i,
                "text": chunk,
                "length": len(chunk),
                "position": i
            })
        print(f"✅ Total chunks created: {len(chunks_with_meta)}")
        return chunks_with_meta