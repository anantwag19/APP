from flask import Flask, request, jsonify
import os
import cohere
import pinecone
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
import re
import configparser  # Import configparser
from pyngrok import ngrok, conf # Import ngrok
from pinecone import Pinecone


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'conda'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
pinecone_api_key = config['pinecone']['api_key']  # Get API key from config
pinecone_environment = config['pinecone']['environment']  # Get environment
pinecone_index_name = "ant"  #  Get index name from config.  You can also put this in the ini


# 1. Initialize Pinecone
# 1. Initialize Pinecone
def initialize_pinecone(api_key: str, environment: str, index_name: str) -> pinecone.Index:
    """Initializes Pinecone and returns the index."""
    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=api_key, environment=environment) # Pass the environment
    # Connect to the index
    try:
        index = pc.Index(index_name)
        pc = Pinecone(api_key=api_key)

        index = pc.Index("pinecode")
    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")
        raise
    print("Pinecone initialized successfully.")

    return index
# 2. Prepare Data for Pinecone
def prepare_data(embeddings: List[List[float]], chunks: List[str], metadata_keys: List[str], metadata_values: List[List[str]]) -> List[Dict]:
    """
    Prepares the data to be inserted into Pinecone.

    Args:
        embeddings: A list of embeddings (each embedding is a list of floats).
        chunks: A list of the original text chunks.
        metadata_keys: A list of metadata keys (e.g., ["chunk_id", "source"]).
        metadata_values: A list of lists, where each inner list contains the metadata
                         values for the corresponding chunk, in the same order as metadata_keys.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a vector
                      and its associated data in the format Pinecone expects.
    """
    if not (len(embeddings) == len(chunks) == len(metadata_values)):
        raise ValueError("Lengths of embeddings, chunks, and metadata_values must be equal.")

    if not metadata_keys:
        pinecone_data = [
            {"id": str(i), "values": embedding, "metadata": {"chunk": chunk}}
            for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
    elif len(metadata_keys) != len(metadata_values[0]):
        raise ValueError("Number of metadata keys must match the number of values in each metadata list.")
    else:
        pinecone_data = []
        for i, (embedding, chunk, metadata_row) in enumerate(zip(embeddings, chunks, metadata_values)):
            metadata = {"chunk": chunk}
            for key, value in zip(metadata_keys, metadata_row):
                metadata[key] = value
            pinecone_data.append({"id": str(i), "values": embedding, "metadata": metadata})
    return pinecone_data

# 3. Insert Data into Pinecone
def insert_data(index: pinecone.Index, data: List[Dict], batch_size: int = 100) -> None:
    """Inserts data into Pinecone in batches."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        index.upsert(vectors=batch)
    print(f"Inserted {len(data)} vectors into Pinecone.")


def get_cohere_embeddings(texts: List[str], cohere_api_key: str, model: str = "embed-english-v3.0") -> List[List[float]]:
    """Generates embeddings for a list of texts using Cohere."""
    co = cohere.Client(cohere_api_key)
    embeddings = co.embed(
        texts=texts,
        model=model
    ).tolist()
    return embeddings



@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(f"****** {filename} ********** Saved on Local")
        return jsonify({'message': 'File uploaded successfully. We are processing !!', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/ingestion', methods=['POST'])
def start_ingestion_pipeline():
    """
    Handles the ingestion pipeline:
    - Loads and chunks the document.
    - Gets embeddings using Cohere.
    - Inserts data into Pinecone.
    """
    # Function to clean text
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.replace('........', '')  # Remove "........"
        text = re.sub(r'•+', '', text)  # Remove sequences of "•"
        return text

    # Function to create overlapping chunks based on words
    def create_word_chunks(text, chunk_size=1024, overlap=100):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    try:
      
        filename = request.json.get('filename')
        print(filename)
        cohere_api_key = config['cohere']['key'] # get key from config
      
        # 1. Initialize Pinecone
        pinecone_index = initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name)
        print("*********Pinecone Initialized**********"+ pinecone_index)
        # Load and process the document
        loader = PyPDFLoader(filename)
        pages = loader.load_and_split()
        full_text = "\n".join([clean_text(page.page_content) for page in pages])
        chunks = create_word_chunks(full_text)

        # 2. Get Cohere Embeddings
        embeddings = get_cohere_embeddings(chunks, cohere_api_key)

        # 3. Prepare Data for Pinecone
        metadata_keys = ["chunk_id", "filename"]  # Define your metadata keys
        metadata_values = [[i, filename] for i in range(len(chunks))]  # Create metadata values
        data = prepare_data(embeddings, chunks, metadata_keys, metadata_values)

        # 4. Insert Data into Pinecone
        insert_data(pinecone_index, data)

        return jsonify({'message': 'Successfully Inserted Data in Pinecone, Please go ahead and ask your questions!'})

    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/query', methods=['POST'])
def queryEngine():
    """Handles querying Pinecone and generating a response using Cohere."""
    def return_results(results, documents):
        Final_Docs = []
        for idx, result in enumerate(results.results):
            Final_Docs.append({str(idx+1): documents[result.index]})
        return Final_Docs

    try:
        cohere_api_key = config['cohere']['key'] # get key from config
        co = cohere.Client(cohere_api_key)
        Final_Docs = []
        rerank_docs = []

        request_data = request.get_json()
        query = request_data.get("query", "")
        print(query)

        # Initialize Pinecone
        pinecone_index = initialize_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name)

        # Generate query embedding
        read_embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key,
            model_name="embed-english-v3.0",
            input_type="search_query",
        )
        query_embedding = read_embed_model.get_text_embedding(query)

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        # Process results
        for match in results.matches:
            chunk_id = match.metadata['chunk_id']
            chunk_data = match.metadata['text']
            content = f"{chunk_id}: {chunk_data}"
            rerank_docs.append(content)
            print(f"*********{chunk_id}**********")
            print(f"*********{chunk_data}**********")

        print("######ReRanking of Document Started#####")
        results = co.rerank(query=query, documents=rerank_docs, top_n=3, model='rerank-english-v2.0')
        Final_Docs = return_results(results, rerank_docs)

        print("***************** Sending to Cohere Finally ********************")
        print(Final_Docs)

        response = co.chat(
            message=query,
            documents=Final_Docs,
            model="command-r-plus",
            temperature=0.3,
        )

        print("**********************O/P************")
        print(response.text)

        return jsonify({'answer': response.text})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    conf.get_default().auth_token = "2cM7k6obd9T9I2f304jKzRychJM_5qjH5v3ciJrAYhM5UWmc7"
    public_url = ngrok.connect(4000)
    print(f"🌍 Public URL: {public_url}")
    # Configure Flask to accept external host (ngrok domain)
    os.environ["FLASK_RUN_HOST"] = "localhost"
    os.environ["FLASK_RUN_PORT"] = "4000"
    app.run(host="localhost", port=4000)
