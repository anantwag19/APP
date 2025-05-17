import os
import cohere
import pinecone
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
import re
import configparser
from pinecone import Pinecone
import tiktoken  # For token counting

# Configuration
print("[INIT] Setting up configuration...")
UPLOAD_FOLDER = 'conda'
print(f"[CONFIG] Upload folder set to: {UPLOAD_FOLDER}")

# Load configuration
print("[CONFIG] Loading config.ini...")
config = configparser.ConfigParser()
config.read('config.ini')
pinecone_api_key = config['pinecone']['api_key']
pinecone_environment = config['pinecone']['environment']
pinecone_index_name = "ant"
cohere_api_key = config['cohere']['key']
print("[CONFIG] API keys loaded successfully")

# Initialize clients
print("[INIT] Initializing clients...")
co = cohere.Client(cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
print("[CLIENTS] Cohere and Pinecone clients initialized")

# Tokenizer init
print("[INIT] Setting up tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 4096
QUERY_TOKENS_BUDGET = 600
CONTEXT_TOKENS_BUDGET = MAX_TOKENS - QUERY_TOKENS_BUDGET
print(f"[TOKENIZER] Ready (Max tokens: {MAX_TOKENS}, Context budget: {CONTEXT_TOKENS_BUDGET})")

def initialize_pinecone() -> pinecone.Index:
    """Initialize Pinecone index."""
    print("[PINECONE] Initializing Pinecone connection...")
    try:
        index = pc.Index(pinecone_index_name)
        print(f"[PINECONE] Successfully connected to index '{pinecone_index_name}'")
        return index
    except Exception as e:
        print(f"[ERROR] Pinecone connection failed: {e}")
        raise

def clean_text(text: str) -> str:
    """Clean text by removing unwanted patterns."""
    print("[CLEANING] Processing text...")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('........', '')
    text = re.sub(r'•+', '', text)
    print("[CLEANING] Text cleaned successfully")
    return text.strip()

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Create text chunks with token limits."""
    print(f"[CHUNKING] Creating chunks (size: {chunk_size}, overlap: {overlap})...")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        # Ensure chunk doesn't exceed token limit
        while count_tokens(chunk) > 500 and len(chunk) > 10:
            chunk = " ".join(chunk.split()[:-20])
        chunks.append(chunk)
        start += chunk_size - overlap
    print(f"[CHUNKING] Created {len(chunks)} chunks")
    return chunks

def get_cohere_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Cohere."""
    print(f"[EMBEDDING] Generating embeddings for {len(texts)} chunks...")
    try:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        print("[EMBEDDING] Embeddings generated successfully")
        return response.embeddings
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        raise

def process_pdf(file_path: str) -> List[str]:
    """Process PDF into clean chunks."""
    print(f"[PDF] Processing file: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(f"[PDF] Loaded {len(pages)} pages")
    full_text = "\n".join([clean_text(page.page_content) for page in pages])
    print("[PDF] Text extracted and cleaned")
    return create_chunks(full_text)

def prepare_data(embeddings: List[List[float]], chunks: List[str], filename: str) -> List[Dict]:
    """Prepare data for Pinecone insertion."""
    print("[PREPARE] Formatting data for Pinecone...")
    data = [{
        "id": f"{filename}_{i}",
        "values": embedding,
        "metadata": {
            "text": chunk,
            "chunk_id": str(i),
            "filename": filename,
            "tokens": count_tokens(chunk)
        }
    } for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))]
    print(f"[PREPARE] Prepared {len(data)} vectors")
    return data

def insert_data(index: pinecone.Index, data: List[Dict]) -> None:
    """Insert data into Pinecone."""
    print("[PINECONE] Starting data insertion...")
    try:
        index.upsert(vectors=data)
        print(f"[PINECONE] Successfully inserted {len(data)} vectors")
    except Exception as e:
        print(f"[ERROR] Insert failed: {e}")
        raise

def start_ingestion(folder_path: str) -> Dict:
    """Main ingestion pipeline."""
    print(f"\n[INGESTION] Starting ingestion from {folder_path}")
    try:
        # Find PDF
        print("[INGESTION] Searching for PDF files...")
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print("[INGESTION] No PDF found")
            return {'error': 'No PDF found'}
        if len(pdf_files) > 1:
            print("[INGESTION] Multiple PDFs found")
            return {'error': 'Multiple PDFs found'}
        
        pdf_path = os.path.join(folder_path, pdf_files[0])
        print(f"[INGESTION] Processing file: {pdf_files[0]}")
        
        chunks = process_pdf(pdf_path)
        embeddings = get_cohere_embeddings(chunks)
        
        index = initialize_pinecone()
        data = prepare_data(embeddings, chunks, pdf_files[0])
        insert_data(index, data)
        
        print("[INGESTION] Completed successfully")
        return {'message': 'Ingestion successful', 'chunks': len(chunks)}
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        return {'error': str(e)}

def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return len(tokenizer.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens."""
    print(f"[TRUNCATE] Truncating text to {max_tokens} tokens")
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

def query_index(query: str) -> Dict:
    """Query Pinecone and generate response from Cohere."""
    print(f"\n[QUERY] Processing query: '{query}'")
    try:
        # Count query tokens and calculate remaining budget
        query_tokens = count_tokens(query)
        remaining_budget = CONTEXT_TOKENS_BUDGET - query_tokens - 200
        print(f"[QUERY] Token budget - Query: {query_tokens}, Remaining: {remaining_budget}")
        
        if remaining_budget <= 0:
            print("[QUERY] Error: Query too long")
            return {'error': 'Query too long - please shorten your query'}

        # Get query embedding
        print("[QUERY] Generating query embedding...")
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = embed_response.embeddings[0]
        print("[QUERY] Embedding generated")
        
        # Query Pinecone
        print("[QUERY] Querying Pinecone index...")
        index = initialize_pinecone()
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        print(f"[QUERY] Found {len(results.matches)} matches")
        
        # Process documents
        print("[QUERY] Processing context documents...")
        context_docs = []
        current_tokens = 0
        for i, match in enumerate(results.matches):
            doc_text = match.metadata['text']
            doc_tokens = count_tokens(doc_text)
            print(f"[QUERY] Match {i+1}: {doc_tokens} tokens")
            
            if doc_tokens > remaining_budget:
                print(f"[QUERY] Skipping match {i+1} (too large)")
                continue
                
            if current_tokens + doc_tokens > remaining_budget:
                available_tokens = remaining_budget - current_tokens
                if available_tokens > 50:
                    print(f"[QUERY] Truncating match {i+1} to fit budget")
                    doc_text = truncate_by_tokens(doc_text, available_tokens)
                    doc_tokens = count_tokens(doc_text)
                else:
                    print("[QUERY] Budget exhausted, stopping")
                    break
                    
            context_docs.append(doc_text)
            current_tokens += doc_tokens
            print(f"[QUERY] Added match {i+1} (Total tokens: {current_tokens})")
            
            if current_tokens >= remaining_budget * 0.9:
                print("[QUERY] Reached 90% of budget, stopping")
                break

        print(f"[QUERY] Final context: {len(context_docs)} docs, {current_tokens} tokens")
        
        # Generate response
        print("[QUERY] Generating response from Cohere...")
        chat_response = co.chat(
            message=query,
            documents=[{"text": doc} for doc in context_docs],
            model="command",
            temperature=0.3,
            max_tokens=min(200, MAX_TOKENS - query_tokens - current_tokens)
        )
        print("[QUERY] Response generated successfully")
        
        return {
            'answer': chat_response.text,
            'contexts': context_docs[:3],
            'token_usage': {
                'query': query_tokens,
                'context': current_tokens,
                'response': count_tokens(chat_response.text)
            }
        }
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    test_folder = "conda"
    test_query = "What is this top 3 integrations in mshs project?"
    
    print("\n=== STARTING PROCESS ===")
    print("[MAIN] Starting ingestion...")
    ingest_result = start_ingestion(test_folder)
    print("[MAIN] Ingestion result:", ingest_result)
    
    if "error" not in ingest_result:
        print("\n[MAIN] Running query...")
        query_result = query_index(test_query)
        print("[MAIN] Query result:", query_result)
        
        if "error" in query_result:
            print("\n[MAIN] Trying with shorter query...")
            shorter_query = test_query[:100]
            query_result = query_index(shorter_query)
            print("[MAIN] Retry result:", query_result)
    print("\n=== PROCESS COMPLETED ===")