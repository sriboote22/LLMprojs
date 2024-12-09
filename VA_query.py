import streamlit as st
import speech_recognition as sr
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceApi
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import os

# Set up Hugging Face LLM and sentence transformer
llm = HuggingFaceHub(huggingfacehub_api_token="HF key", repo_id="google/flan-t5-large")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# Initialize Pinecone
pc = Pinecone(api_key="your Pinecone Key")
index_name = "vaquery"
index = pc.Index(index_name)

# Initialize recognizer
recognizer = sr.Recognizer()

# Streamlit UI
st.title("Voice-based Query App")
st.write("Upload an audio file or record your query to get summarized answers from your documents.")

# Option to upload an audio file or record a query
audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
if st.button("Record Audio"):
    st.warning("Voice recording is not natively supported in Streamlit. You may use the file upload option instead.")

def read_document(dir):
    file_loader = PyPDFDirectoryLoader(dir)
    doc = file_loader.load()
    return doc

def chunk_data(doc, chunk_size=700, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(doc)
    return doc

def get_embeddings(texts):
    return model.encode(texts)

def retrieve_query(query, k=2):
    query_embedding = get_embeddings([query])[0]
    results = index.query(vector=query_embedding.tolist(), top_k=k, include_metadata=True)
    return results['matches']

def retrieve_answers(query):
    doc_search = retrieve_query(query)
    documents_content = [
        Document(page_content=match['metadata']['text'], metadata={"source": f"chunk-{match['id']}"})
        for match in doc_search
    ]
    response = chain.run(input_documents=documents_content, question=f"Summarize answer for: {query}")
    return response

def speech_to_text_from_file(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, the audio was not clear enough to transcribe.")
        except sr.RequestError:
            st.error("Sorry, the speech service is currently unavailable.")

# Process audio file
if audio_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    recognized_text = speech_to_text_from_file("temp_audio.wav")
    if recognized_text:
        st.write("Recognized Text:", recognized_text)
        with st.spinner("Querying documents..."):
            answer = retrieve_answers(recognized_text)
        st.write("Answer:", answer)
    else:
        st.write("Could not transcribe the audio.")
