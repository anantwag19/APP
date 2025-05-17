import os
import cohere
import pinecone
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
import re
import configparser
from pinecone import Pinecone
import tiktoken

# Configuration
UPLOAD_FOLDER = 'conda'
print(f"Configuration: Using upload folder '{UPLOAD_FOLDER}'")

# Load configuration
print("\nLoading configuration...")
config = configparser.ConfigParser()
config.read('config.ini')
pinecone_api_key = config['pinecone']['api_key']
pinecone_environment = config['pinecone']['environment']
pinecone_index_name = "ant"
cohere_api_key = config['cohere']['key']
print("Configuration loaded successfully")

# Initialize clients
print("\nInitializing clients...")
co = cohere.Client(cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
print("Clients initialized (Cohere and Pinecone)")

# Tokenizer setup
print("\nInitializing tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 4096
QUERY_TOKENS_BUDGET = 600
CONTEXT_TOKENS_BUDGET = MAX_TOKENS - QUERY_TOKENS_BUDGET
print(f"Tokenizer ready | Max tokens: {MAX_TOKENS} | Context budget: {CONTEXT_TOKENS_BUDGET}")

def initialize_pinecone() -> pinecone.Index:
    """Initialize Pinecone index with verbose logging."""
    print("\nInitializing Pinecone connection...")
    try:
        index = pc.Index(pinecone_index_name)
        print(f"✓ Pinecone connected to index '{pinecone_index_name}'")
        return index
    except Exception as e:
        print(f"✗ Pinecone connection failed: {e}")
        raise

def clean_text(text: str) -> str:
    """Clean text with progress reporting."""
    print("Cleaning text...", end=" ")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('........', '')
    text = re.sub(r'•+', '', text)
    print("Done")
    return text.strip()

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Create chunks with progress tracking."""
    print(f"\nCreating chunks (size: {chunk_size}, overlap: {overlap})...")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        
        # Token limit enforcement
        while count_tokens(chunk) > 500 and len(chunk) > 10:
            chunk = " ".join(chunk.split()[:-20])
            
        chunks.append(chunk)
        start += chunk_size - overlap
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def get_cohere_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings with progress tracking."""
    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    try:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        print("✓ Embeddings generated successfully")
        return response.embeddings
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        raise

def process_pdf(file_path: str) -> List[str]:
    """Process PDF with step-by-step logging."""
    print(f"\nProcessing PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    print("- Loading pages...", end=" ")
    pages = loader.load_and_split()
    print(f"Loaded {len(pages)} pages")
    
    print("- Cleaning text...", end=" ")
    full_text = "\n".join([clean_text(page.page_content) for page in pages])
    print("Done")
    
    return create_chunks(full_text)

def prepare_data(embeddings: List[List[float]], chunks: List[str], filename: str) -> List[Dict]:
    """Prepare data with progress tracking."""
    print(f"\nPreparing {len(chunks)} vectors for Pinecone...")
    return [{
        "id": f"{filename}_{i}",
        "values": embedding,
        "metadata": {
            "text": chunk,
            "chunk_id": str(i),
            "filename": filename,
            "tokens": count_tokens(chunk)
        }
    } for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))]

def insert_data(index: pinecone.Index, data: List[Dict]) -> None:
    """Insert data with verbose logging."""
    print(f"\nInserting {len(data)} vectors into Pinecone...")
    try:
        index.upsert(vectors=data)
        print("✓ Data inserted successfully")
    except Exception as e:
        print(f"✗ Insertion failed: {e}")
        raise

def start_ingestion(folder_path: str) -> Dict:
    """Main ingestion pipeline with detailed logging."""
    print("\n" + "="*50)
    print("Starting Ingestion Pipeline")
    print("="*50)
    
    try:
        # Find PDF
        print("\nSearching for PDF files...")
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("✗ No PDF found")
            return {'error': 'No PDF found'}
        if len(pdf_files) > 1:
            print("✗ Multiple PDFs found")
            return {'error': 'Multiple PDFs found'}
        
        pdf_path = os.path.join(folder_path, pdf_files[0])
        print(f"✓ Found PDF: {pdf_files[0]}")
        
        # Process PDF
        chunks = process_pdf(pdf_path)
        embeddings = get_cohere_embeddings(chunks)
        
        # Pinecone operations
        index = initialize_pinecone()
        data = prepare_data(embeddings, chunks, pdf_files[0])
        insert_data(index, data)
        
        print("\n" + "="*50)
        print("Ingestion Complete")
        print("="*50)
        return {'message': 'Ingestion successful', 'chunks': len(chunks)}
    except Exception as e:
        print(f"\n✗ Ingestion failed: {str(e)}")
        return {'error': str(e)}

def count_tokens(text: str) -> int:
    """Count tokens silently (no print to avoid spam)."""
    return len(tokenizer.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text with token limits."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

def query_index(query: str) -> Dict:
    """Query Pinecone and generate response with top 3 reranked chunks."""
    try:
        # Count query tokens and calculate remaining budget
        query_tokens = count_tokens(query)
        remaining_budget = CONTEXT_TOKENS_BUDGET - query_tokens - 200
        
        if remaining_budget <= 0:
            return {'error': 'Query too long - please shorten your query'}

        # Get query embedding
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = embed_response.embeddings[0]
        
        # Query Pinecone
        index = initialize_pinecone()
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        # Check if we got any results
        if not results.matches:
            print("No documents found matching the query")
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'contexts': [],
                'token_usage': {
                    'query': query_tokens,
                    'context': 0,
                    'response': 0
                }
            }
        
        # Extract documents for reranking
        candidate_docs = [match.metadata['text'] for match in results.matches]
        original_scores = [match.score for match in results.matches]
        
        # Print original chunks
        print("\n" + "="*50)
        print("ORIGINAL CHUNKS (BEFORE RERANKING):")
        print("="*50)
        for i, (doc, score) in enumerate(zip(candidate_docs, original_scores)):
            print(f"\nCHUNK #{i+1} (Pinecone Score: {score:.4f}):")
            print("-"*40)
            print(doc[:500] + "..." if len(doc) > 500 else doc)
            print("-"*40)
        
        # Rerank to get top 3 most relevant chunks
        ranked_docs = candidate_docs[:3]  # Default fallback
        
        try:
            rerank_response = co.rerank(
                query=query,
                documents=candidate_docs,
                top_n=3,  # Get exactly top 3 most relevant
                model="rerank-english-v3.0"
            )
            
            # Print reranked results
            print("\n" + "="*50)
            print("TOP 3 RERANKED CHUNKS:")
            print("="*50)
            for i, result in enumerate(rerank_response.results):
                print(f"\nRANK #{i+1} (Was #{result.index + 1} originally):")
                print(f"Relevance Score: {result.relevance_score:.4f}")
                print("-"*40)
                print(candidate_docs[result.index][:500] + "..." if len(candidate_docs[result.index]) > 500 else candidate_docs[result.index])
                print("-"*40)
            
            ranked_docs = [candidate_docs[result.index] for result in rerank_response.results]
            
        except Exception as e:
            print(f"\nReranking failed: {str(e)}")
            print("Using top 3 unreranked results")
        
        # Process documents with token control
        context_docs = []
        current_tokens = 0
        for doc_text in ranked_docs:
            doc_tokens = count_tokens(doc_text)
            
            if doc_tokens > remaining_budget:
                continue
                
            if current_tokens + doc_tokens > remaining_budget:
                available_tokens = remaining_budget - current_tokens
                if available_tokens > 50:
                    doc_text = truncate_by_tokens(doc_text, available_tokens)
                    doc_tokens = count_tokens(doc_text)
                else:
                    continue
                    
            context_docs.append(doc_text)
            current_tokens += doc_tokens
            
            if current_tokens >= remaining_budget * 0.9:
                break

        print(f"\nToken usage:")
        print(f"- Query: {query_tokens}")
        print(f"- Context: {current_tokens}")
        print(f"- Remaining: {MAX_TOKENS - query_tokens - current_tokens}")

        # Generate response
        chat_response = co.chat(
            message=query,
            documents=[{"text": doc} for doc in context_docs],
            model="command",
            temperature=0.3,
            max_tokens=min(200, MAX_TOKENS - query_tokens - current_tokens)
        )
        
        return {
            'answer': chat_response.text,
            'contexts': context_docs,
            'token_usage': {
                'query': query_tokens,
                'context': current_tokens,
                'response': count_tokens(chat_response.text)
            }
        }
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Main Execution")
    print("="*50)
    
    test_folder = "conda"
    test_query = "What is this document about ?"
    
    print("\n=== INGESTION ===")
    ingest_result = start_ingestion(test_folder)
    print("\nIngestion result:", ingest_result)
    
    if "error" not in ingest_result:
        print("\n=== QUERY ===")
        query_result = query_index(test_query)
        print("\nQuery result:", query_result)
        
        if "error" in query_result:
            print("\n=== RETRY WITH SHORTER QUERY ===")
            shorter_query = test_query[:100]
            query_result = query_index(shorter_query)
            print("\nRetry result:", query_result)
    
    print("\n" + "="*50)
    print("Execution Complete")
    print("="*50)