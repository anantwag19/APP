import streamlit as st
import requests

# Function to get response from the API
def get_api_response(user_input):
    try:
        response = requests.post('https://70c4-27-4-56-158.ngrok-free.app/query', json={'query': user_input})
        response.raise_for_status()
        return response.json().get('answer')
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

# Function to upload file to the API
def upload_file_to_api(uploaded_file):
    try:
        upload_url = 'https://70c4-27-4-56-158.ngrok-free.app/upload'
        files = {'file': uploaded_file}
        response = requests.post(upload_url, files=files)
        response.raise_for_status()
        return response.json().get('message', 'File uploaded successfully.'), response.json().get('filename')
    except requests.exceptions.RequestException as e:
        return f"An error occurred while uploading the file: {str(e)}", None

# Function to call the ingestion web service
def call_ingestion_service(filename):
    try:
        print(filename)
        ingestion_url = 'https://70c4-27-4-56-158.ngrok-free.app/ingestion'
        response = requests.post(ingestion_url, json={'filename': filename})
        response.raise_for_status()
        return response.json().get('message', 'Ingestion successful.')
    except requests.exceptions.RequestException as e:
        return f"An error occurred during ingestion: {str(e)}"

st.title("ChatBOT")

# Display the logo image
st.image("/Users/anantwaghmare/Desktop/app/meilleur-chatbot.jpg", width=150)

# Text input for user questions
user_input = st.text_input("Please ask me questions:")

# Button to submit the question
if st.button("Submit"):
    if user_input:
        with st.spinner('Thinking...'):
            chatbot_response = get_api_response(user_input)
        st.write(f"Chatbot: {chatbot_response}")
    else:
        st.write("Please enter a question.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner('Uploading...'):
        upload_response, filename = upload_file_to_api(uploaded_file)
        st.write(upload_response)
        if 'successfully' in upload_response and filename:
            with st.spinner('Ingesting...'):
                ingestion_response = call_ingestion_service(filename)
            st.write(ingestion_response)
