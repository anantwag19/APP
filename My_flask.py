from flask import Flask, request, jsonify
import os
import cohere
import pinecone
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
import re
import configparser
from pyngrok import ngrok, conf
from pinecone import Pinecone
import tiktoken

app = Flask(__name__)

print("[INIT] Starting application initialization...")

# Configuration
UPLOAD_FOLDER = 'conda'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(f"[CONFIG] Upload folder set to: {UPLOAD_FOLDER}")

# Load configuration
print("[CONFIG] Loading configuration from config.ini...")
config = configparser.ConfigParser()
config.read('config.ini')
pinecone_api_key = config['pinecone']['api_key']
pinecone_environment = config['pinecone']['environment']
pinecone_index_name = "ant"
cohere_api_key = config['cohere']['key']
print("[CONFIG] Configuration loaded successfully")

# Initialize clients
print("[INIT] Initializing Cohere and Pinecone clients...")
co = cohere.Client(cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
print("[INIT] Clients initialized successfully")

# Tokenizer setup
print("[INIT] Setting up tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 4096
QUERY_TOKENS_BUDGET = 600
CONTEXT_TOKENS_BUDGET = MAX_TOKENS - QUERY_TOKENS_BUDGET
print(f"[INIT] Tokenizer ready | Max tokens: {MAX_TOKENS} | Context budget: {CONTEXT_TOKENS_BUDGET}")

def initialize_pinecone() -> pinecone.Index:
    """Initialize Pinecone index."""
    print("[PINECONE] Initializing Pinecone connection...")
    try:
        index = pc.Index(pinecone_index_name)
        print("[PINECONE] Successfully connected to Pinecone index")
        return index
    except Exception as e:
        print(f"[ERROR] Pinecone connection failed: {e}")
        raise

def clean_text(text: str) -> str:
    """Clean text by removing unwanted patterns."""
    print("[PROCESSING] Cleaning text...")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('........', '')
    text = re.sub(r'•+', '', text)
    print("[PROCESSING] Text cleaning completed")
    return text.strip()

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Create text chunks with token limits."""
    print(f"[PROCESSING] Creating chunks (size: {chunk_size}, overlap: {overlap})...")
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
    print(f"[PROCESSING] Created {len(chunks)} chunks")
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
    print(f"[PROCESSING] Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    print("[PROCESSING] Extracting pages...")
    pages = loader.load_and_split()
    print(f"[PROCESSING] Loaded {len(pages)} pages")
    
    print("[PROCESSING] Processing full text...")
    full_text = "\n".join([clean_text(page.page_content) for page in pages])
    chunks = create_chunks(full_text)
    print("[PROCESSING] PDF processing completed")
    return chunks

def prepare_data(embeddings: List[List[float]], chunks: List[str], filename: str) -> List[Dict]:
    """Prepare data for Pinecone insertion."""
    print(f"[PINECONE] Preparing {len(chunks)} vectors for insertion...")
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
    """Insert data into Pinecone."""
    print("[PINECONE] Starting data insertion...")
    try:
        index.upsert(vectors=data)
        print(f"[PINECONE] Successfully inserted {len(data)} vectors")
    except Exception as e:
        print(f"[ERROR] Insertion failed: {e}")
        raise

def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return len(tokenizer.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload."""
    print("\n[API] File upload request received")
    try:
        if 'file' not in request.files:
            print("[ERROR] No file part in request")
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            print("[ERROR] No file selected")
            return jsonify({'error': 'No selected file'})

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"[FILE] Saving file to: {filename}")
        file.save(filename)
        print("[FILE] File saved successfully")
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/ingestion', methods=['POST'])
def start_ingestion():
    """Main ingestion pipeline."""
    print("\n[API] Ingestion request received")
    try:
        filename = request.json.get('filename')
        if not filename:
            print("[ERROR] No filename provided")
            return jsonify({'error': 'No filename provided'})

        print(f"[INGESTION] Processing file: {filename}")
        chunks = process_pdf(filename)
        print("[INGESTION] Generating embeddings...")
        embeddings = get_cohere_embeddings(chunks)
        
        print("[INGESTION] Preparing Pinecone index...")
        index = initialize_pinecone()
        data = prepare_data(embeddings, chunks, os.path.basename(filename))
        insert_data(index, data)
        
        print("[INGESTION] Pipeline completed successfully")
        return jsonify({'message': 'Ingestion successful', 'chunks': len(chunks)})
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/query', methods=['POST'])
def query_index():
    """Query Pinecone and generate response with top 3 reranked chunks."""
    print("\n[API] Query request received")
    try:
        query = request.json.get('query')
        if not query:
            print("[ERROR] No query provided")
            return jsonify({'error': 'No query provided'})

        print("[QUERY] Analyzing query...")
        # Count query tokens and calculate remaining budget
        query_tokens = count_tokens(query)
        remaining_budget = CONTEXT_TOKENS_BUDGET - query_tokens - 200
        
        if remaining_budget <= 0:
            print("[ERROR] Query too long")
            return jsonify({'error': 'Query too long - please shorten your query'})

        print("[QUERY] Generating query embedding...")
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = embed_response.embeddings[0]
        
        print("[QUERY] Querying Pinecone index...")
        index = initialize_pinecone()
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        if not results.matches:
            print("[QUERY] No matching documents found")
            return jsonify({'error': 'No matching documents found'})

        print(f"[QUERY] Found {len(results.matches)} candidate documents")
        # Extract candidate documents
        candidate_docs = [match.metadata['text'] for match in results.matches]
        
        print("[RERANK] Starting reranking process...")
        # Get top 3 reranked documents
        try:
            rerank_response = co.rerank(
                query=query,
                documents=candidate_docs,
                top_n=3,
                model="rerank-english-v3.0"
            )
            top_chunks = [candidate_docs[result.index] for result in rerank_response.results]
            print("[RERANK] Successfully reranked documents")
        except Exception as e:
            print(f"[WARNING] Reranking failed, using Pinecone results: {str(e)}")
            top_chunks = [match.metadata['text'] for match in results.matches[:3]]

        print("[PROCESSING] Applying token limits...")
        # Process chunks with token limits
        context_docs = []
        current_tokens = 0
        for chunk in top_chunks:
            chunk_tokens = count_tokens(chunk)
            if current_tokens + chunk_tokens > remaining_budget:
                available = remaining_budget - current_tokens
                if available > 50:
                    chunk = truncate_by_tokens(chunk, available)
                    chunk_tokens = count_tokens(chunk)
                else:
                    continue
            context_docs.append(chunk)
            current_tokens += chunk_tokens

        print("[RESPONSE] Generating final response...")
        # Generate final response with strict instructions
        strict_prompt = f"""
        [IMPORTANT INSTRUCTIONS]
        1. Only use the information provided in the following context documents
        2. Do not add any external knowledge or information
        3. If the answer isn't found in the documents, respond with "The answer is not contained in the provided information"
        4. Be precise and only reference the given content
        
        Question: {query}
        """
        
        chat_response = co.chat(
            message=strict_prompt,
            documents=[{"text": chunk} for chunk in context_docs],
            model="command",
            temperature=0.3,
            max_tokens=min(200, MAX_TOKENS - query_tokens - current_tokens)
        )
        
        print("[QUERY] Request completed successfully")
        return jsonify({
            'answer': chat_response.text,
            'contexts': context_docs,
            'token_usage': {
                'query': query_tokens,
                'context': current_tokens,
                'response': count_tokens(chat_response.text)
            }
        })
    except Exception as e:
        print(f"[ERROR] Query processing failed: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n[STARTUP] Starting application server...")
    conf.get_default().auth_token = "2cM7k6obd9T9I2f304jKzRychJM_5qjH5v3ciJrAYhM5UWmc7"
    public_url = ngrok.connect(4000)
    print(f"🌍 Public URL: {public_url}")
    os.environ["FLASK_RUN_HOST"] = "localhost"
    os.environ["FLASK_RUN_PORT"] = "4000"
    print("[SERVER] Ready to accept requests")
    app.run(host="localhost", port=4000)