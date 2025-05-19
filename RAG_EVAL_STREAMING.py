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
from llama_cpp import Llama
import numpy as np
import json
import pandas as pd
from tabulate import tabulate
from flask import Flask, request, jsonify, Response  # Add Response here
from flask import stream_with_context  # Also add this for streaming support

app = Flask(__name__)

# =================================================================
# 1. CONFIGURATION (Edit these paths as needed)
# =================================================================
GROUND_TRUTH_PATH = "gold.json"  # Path to your ground truth file
MISTRAL_MODEL_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
UPLOAD_FOLDER = 'conda'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =================================================================
# 2. INITIALIZATION WITH STAGE TRACKING
# =================================================================
print("\n" + "="*60)
print("INITIALIZING APPLICATION".center(60))
print("="*60)

# Load ground truth
print("\n[STAGE] Loading ground truth...")
try:
    with open(GROUND_TRUTH_PATH) as f:
        GROUND_TRUTH_DATA = json.load(f)
    if not isinstance(GROUND_TRUTH_DATA, list):
        raise ValueError("Ground truth data must be a list of Q&A pairs")
    print(f"[SUCCESS] Loaded {len(GROUND_TRUTH_DATA)} evaluation examples")
except Exception as e:
    print(f"[ERROR] Failed to load ground truth: {str(e)}")
    GROUND_TRUTH_DATA = []

# Load config
print("\n[STAGE] Loading config...")
try:
    config = configparser.ConfigParser()
    config.read('config.ini')
    pinecone_api_key = config['pinecone']['api_key']
    cohere_api_key = config['cohere']['key']
    pinecone_index_name = "ant"
    print("[SUCCESS] Config loaded")
except Exception as e:
    print(f"[ERROR] Config error: {str(e)}")
    raise

# Initialize Mistral-7B
print("\n[STAGE] Loading Mistral-7B...")
try:
    eval_llm = Llama(
        model_path=MISTRAL_MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=33
    )
    print("[SUCCESS] Mistral-7B loaded")
except Exception as e:
    print(f"[ERROR] Mistral load failed: {str(e)}")
    raise

# Initialize clients
print("\n[STAGE] Initializing clients...")
try:
    co = cohere.Client(cohere_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    print("[SUCCESS] Cohere and Pinecone ready")
except Exception as e:
    print(f"[ERROR] Client init failed: {str(e)}")
    raise

# Tokenizer setup
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 4096
QUERY_TOKENS_BUDGET = 600
CONTEXT_TOKENS_BUDGET = MAX_TOKENS - QUERY_TOKENS_BUDGET

print("\n" + "="*60)
print("INITIALIZATION COMPLETE".center(60))
print("="*60)

# =================================================================
# 3. CORE RAG FUNCTIONS
# =================================================================
def initialize_pinecone() -> pinecone.Index:
    print("\n[PINECONE] Connecting to index...")
    try:
        index = pc.Index(pinecone_index_name)
        print(f"[SUCCESS] Connected to index '{pinecone_index_name}'")
        return index
    except Exception as e:
        print(f"[ERROR] Pinecone error: {str(e)}")
        raise

def process_pdf(file_path: str) -> List[str]:
    print(f"\n[PROCESSING] Loading PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        full_text = "\n".join([re.sub(r'\s+', ' ', p.page_content) for p in pages])
        chunks = create_chunks(full_text)
        print(f"[SUCCESS] Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"[ERROR] PDF processing failed: {str(e)}")
        raise

def create_chunks(text: str, chunk_size=500, overlap=100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        while count_tokens(chunk) > 500 and len(chunk) > 10:
            chunk = " ".join(chunk.split()[:-20])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text

# =================================================================
# 4. EVALUATION SYSTEM (YOUR EXACT LOGIC)
# =================================================================
def generate_mistral_answer(question: str, contexts: List[str]) -> str:
    print("\n[EVALUATION] Generating Mistral answer...")
    try:
        context_str = "\n".join(contexts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        output = eval_llm(prompt, max_tokens=50, temperature=0.1)
        answer = output["choices"][0]["text"].strip()
        print(f"[SUCCESS] Answer: {answer[:100]}...")
        return answer
    except Exception as e:
        print(f"[ERROR] Mistral failed: {str(e)}")
        raise

def evaluate_response(question: str, contexts: List[str], answer: str) -> Dict:
    print("\n[EVALUATION] Calculating metrics...")
    ground_truth = next(
        (item["ground_truth"] for item in GROUND_TRUTH_DATA 
         if item["question"].lower() == question.lower()), 
        None
    )
    
    if not ground_truth:
        print("[WARNING] No ground truth - skipping evaluation")
        return None

    # Your exact evaluation logic
    relevant_contexts = sum(1 for ctx in contexts if any(gt.lower() in ctx.lower() for gt in ground_truth))
    metrics = {
        "context_precision": round(relevant_contexts / len(contexts), 2),
        "context_recall": 1.0 if relevant_contexts > 0 else 0.0,
        "answer_relevancy": 1.0 if any(gt.lower() in answer.lower() for gt in ground_truth) else 0.0,
        "faithfulness": int(all(not any(f"not {gt}" in ctx.lower() for ctx in contexts) for gt in ground_truth)),
        "ground_truth": ground_truth
    }
    
    # Print evaluation matrix (your exact format)
    print("\n" + "="*60)
    print("RAG EVALUATION MATRIX".center(60))
    print("="*60)
    print(f"\nQuestion: {question}")
    print(f"Model Answer: {answer}")
    print(f"Ground Truth: {ground_truth}")
    print("\n📊 Metrics:")
    print(f"Context Precision: {metrics['context_precision']:.2f}")
    print(f"Context Recall: {metrics['context_recall']:.2f}")
    print(f"Answer Relevancy: {metrics['answer_relevancy']}")
    print(f"Faithfulness: {metrics['faithfulness']}")
    print("="*60)
    
    return metrics

# =================================================================
# 5. FLASK ENDPOINTS (WITH RERANKING)
# =================================================================
@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n" + "="*60)
    print("FILE UPLOAD".center(60))
    print("="*60)
    try:
        if 'file' not in request.files:
            raise ValueError("No file part")
            
        file = request.files['file']
        if file.filename == '':
            raise ValueError("No selected file")

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(f"[SUCCESS] Saved to {filename}")
        return jsonify({'message': 'File uploaded', 'filename': filename})
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/ingestion', methods=['POST'])
def start_ingestion():
    print("\n" + "="*60)
    print("INGESTION STARTED".center(60))
    print("="*60)
    try:
        filename = request.json.get('filename')
        if not filename:
            raise ValueError("Missing filename")
        
        chunks = process_pdf(filename)
        embeddings = co.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        
        index = initialize_pinecone()
        index.upsert(vectors=[
            {
                "id": f"{os.path.basename(filename)}_{i}",
                "values": emb,
                "metadata": {"text": chunk}
            } for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
        ])
        print(f"[SUCCESS] Ingested {len(chunks)} chunks")
        return jsonify({'message': 'Ingestion complete', 'chunks': len(chunks)})
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
 

@app.route('/query_stream', methods=['POST'])
def query_stream():
    try:
        print("\n=== [DEBUG] STARTING QUERY STREAM ===")
        query = request.json.get('query')
        if not query:
            raise ValueError("Empty query")

        print(f"\n📩 QUERY RECEIVED: {query}")

        # 1. Retrieve and Rerank Contexts
        try:
            print("\n🔍 STEP 1: RETRIEVING FROM PINECONE")
            query_embed = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings[0]
            index = initialize_pinecone()
            pinecone_results = index.query(vector=query_embed, top_k=10, include_metadata=True)
            
            # Print raw Pinecone results
            print("\n📦 RAW PINECONE RESULTS (BEFORE RERANKING):")
            for i, match in enumerate(pinecone_results.matches):
                print(f"\n🔹 RESULT {i+1}:")
                print(f"   ID: {match.id}")
                print(f"   Score: {match.score:.3f}")
                print(f"   Text: {match.metadata['text'][:150]}...")

            candidate_docs = [m.metadata['text'] for m in pinecone_results.matches]
            
            print("\n⚖️ STEP 2: RERANKING WITH COHERE")
            rerank_results = co.rerank(query=query, documents=candidate_docs, top_n=2, model="rerank-english-v3.0")
            
            # Print reranking results
            print("\n🎯 RERANKING RESULTS:")
            for i, result in enumerate(rerank_results.results):
                print(f"\n🏅 RANK {i+1}:")
                print(f"   Original Index: {result.index}")
                print(f"   Relevance Score: {result.relevance_score:.3f}")
                print(f"   Text: {candidate_docs[result.index][:150]}...")

            contexts = [candidate_docs[r.index] for r in rerank_results.results]
            
            # Print final contexts being sent to Cohere
            print("\n🚀 FINAL CONTEXTS SENT TO COHERE CHAT MODEL:")
            for i, ctx in enumerate(contexts):
                print(f"\n📜 CONTEXT {i+1} ({len(ctx.split())} words):")
                print(f"{ctx[:200]}...")

        except Exception as retrieval_error:
            print(f"\n❌ RETRIEVAL ERROR: {str(retrieval_error)}")
            raise

        # 2. Stream answer with evaluation
        def generate():
            full_answer = []
            
            if not contexts:
                print("\n⚠️ NO CONTEXTS FOUND FOR QUERY")
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant contexts found'})}\n\n"
                return

            try:
                # Print the exact prompt being sent to Cohere
                cohere_prompt = f"Answer strictly from these contexts:\n\n{contexts}\n\nQuestion: {query}"
                print("\n💬 COHERE PROMPT BEING SENT:")
                print(cohere_prompt[:500] + "...")

                cohere_stream = co.chat_stream(
                    message=cohere_prompt,
                    documents=[{"text": c} for c in contexts],
                    model="command",
                    temperature=0.1,
                    preamble="""preamble = "Answer from these contexts only. If unsure, say 'Not in documents'.""."""
                )

                for event in cohere_stream:
                    if event.event_type == "text-generation":
                        chunk = event.text
                        full_answer.append(chunk)
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

                full_answer_str = "".join(full_answer)
                print("\n🤖 FINAL ANSWER GENERATED:")
                print(full_answer_str)

                evaluation = evaluate_response(query, contexts, full_answer_str)
                
                # Print evaluation results
                print("\n🧐 EVALUATION RESULTS:")
                print(f"Relevance: {evaluation['relevance']}")
                print(f"Grounding: {evaluation['grounding']}")
                print(f"Usefulness: {evaluation['usefulness']}")
                
                yield f"data: {json.dumps({'type': 'evaluation', 'data': evaluation})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as generation_error:
                print(f"\n❌ GENERATION ERROR: {str(generation_error)}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(generation_error)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        print(f"\n❌ ENDPOINT ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_index():
    print("\n" + "="*60)
    print("QUERY PROCESSING".center(60))
    print("="*60)
    try:
        # 1. Validate query
        query = request.json.get('query')
        if not query:
            raise ValueError("Empty query")
        
        print(f"[QUERY] '{query[:50]}...'")
        query_tokens = count_tokens(query)
        if query_tokens > QUERY_TOKENS_BUDGET:
            raise ValueError(f"Query too long ({query_tokens} tokens)")

        # 2. Retrieve contexts
        print("\n[RETRIEVAL] Getting contexts...")
        query_embed = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]
        
        index = initialize_pinecone()
        pinecone_results = index.query(
            vector=query_embed,
            top_k=10,
            include_metadata=True
        )
        
        if not pinecone_results.matches:
            raise ValueError("No results from Pinecone")
        
        # 3. Rerank (your original logic)
        print("\n[RERANKING] Running Cohere reranker...")
        candidate_docs = [m.metadata['text'] for m in pinecone_results.matches]
        rerank_results = co.rerank(
            query=query,
            documents=candidate_docs,
            top_n=3,
            model="rerank-english-v3.0"
        )
        contexts = [candidate_docs[r.index] for r in rerank_results.results]
        print(f"[SUCCESS] Selected top {len(contexts)} contexts")

        # 4. Generate answers
        print("\n[GENERATION] Creating Cohere answer...")
        cohere_answer = co.chat(
            message=query,
            documents=[{"text": c} for c in contexts],
            model="command",
            temperature=0.3
        ).text
        print(f"[ANSWER] {cohere_answer[:100]}...")

        # 5. Evaluate with Mistral
        print("\n[EVALUATION] Starting evaluation pipeline...")
        mistral_answer = generate_mistral_answer(query, contexts)
        evaluation = evaluate_response(query, contexts, mistral_answer)
        
        # 6. Prepare response
        response = {
            'answer': cohere_answer,
            'contexts': contexts,
            'evaluation': evaluation
        }
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =================================================================
# 6. START SERVER
# =================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("SERVER STARTING".center(60))
    print("="*60)
    try:
        conf.get_default().auth_token = "2cM7k6obd9T9I2f304jKzRychJM_5qjH5v3ciJrAYhM5UWmc7"
        public_url = ngrok.connect(4000)
        print(f"\n🌐 Public URL: {public_url}")
        print("🛠️  Endpoints:")
        print("- POST /upload : Upload PDF")
        print("- POST /ingestion : Process PDF")
        print("- POST /query : Ask questions")
        print("\n[READY] Server running on port 4000")
        app.run(host="localhost", port=4000)
    except Exception as e:
        print(f"[FATAL] Server failed: {str(e)}")