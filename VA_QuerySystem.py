
import speech_recognition as sr

#required libraries: Hugginface, unstructured,tiktoken,pinecode-client, pypdf, langchain,pandas, numpy,python-dotenv

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import pinecone
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embedding.openai import OpenAIEmbeddings
from huggingface_hub import InferenceApi
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
#from langchain.llms import HuggingFace

import os


# Set up Hugging Face LLM (you can also load models locally if required)
# Replace "model_id" with any model available on Hugging Face Hub (e.g., "gpt2" or "distilgpt2")
llm = HuggingFaceHub(huggingfacehub_api_token="hf_key",repo_id="meta-llama/Llama-3.2-1B")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
chain = load_qa_chain(llm, chain_type="stuff")

def read_document(dir):
    file_loader=PyPDFDirectoryLoader(dir)
    doc=file_loader.load()
    return doc

def chunk_data(doc,chunk_size=700,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc =text_splitter.split_documents(doc)
    return doc

def get_embeddings(texts):
    return model.encode(texts)

doc = read_document('documents/')
print(len(doc))


documents = chunk_data(doc=doc)
#print(documents)
print(len(documents))
embeddings = get_embeddings([doc.page_content for doc in documents])
print(embeddings.shape)

pc = Pinecone(
        api_key="your Pinecone key"
    )


index_name = "vaquery"
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=embeddings.shape[1])  # Specify dimension based on embedding size
index = pc.Index(index_name)
#vectorstore = pc.from_documents(doc,embeddings,index_name=index_name)

for i, doc in enumerate(documents):
    metadata = {"text": doc.page_content, "chunk": i}
    index.upsert([(f'doc-{i}', embeddings[i].tolist(),metadata)])

def retrieve_query(query, k=4):
    query_embedding = get_embeddings([query])[0]  # Get embedding for the query
    # Use keyword arguments for the query method
    results = index.query(vector=query_embedding.tolist(), top_k=k, include_metadata=True)
    return results['matches']

def retrieve_answers(query):
    doc_search = retrieve_query(query)
    documents_content = [
        Document(page_content=match['metadata']['text'], metadata={"source": f"chunk-{match['id']}"})
        for match in doc_search
    ]  # Wrap matched document text in a Document object with page_content and metadata

    #print("Matched Documents:", [doc.page_content for doc in documents_content])  # Print matched document content
    for doc in doc_search:
        print(f"Document ID: {doc['id']}, Content: {doc['metadata']['text']}")

    detailed_prompt = f"""
    You are provided with several chunks of information from documents. Answer the following question based solely on this information.
    
    Question: {query}
    
    Only respond based on the content provided in the following documents:
    {''.join([doc.page_content for doc in documents_content])}
    """
    response = chain.run(input_documents=documents_content, question=detailed_prompt)
    # print(doc_search)  # Print IDs of matched documents
    # response = chain.run(input_documents=doc_search, question=query)
    return response

#our_query = "Who is Sriparna?"
#answer = retrieve_answers(our_query)
#print(answer)

recognizer = sr.Recognizer()

#wav file
audio_file_path = "G:\OpenCV_Course\python_env\LLMPrompts\VAR.wav"
# import os
# audio_file_path = os.path.join("G:", "OpenCV_Course", "python_env", "LLMPROMPTS", "results.wav")


def speech_to_text_from_file(file_path):
    with sr.AudioFile(file_path) as source:
        
        audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            print("The audio file says: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, the speech service is unavailable.")
            return None
        
#Final call

recognized_text = speech_to_text_from_file(audio_file_path)
#recognized_text = "Interpolation"

if recognized_text:
    answer = retrieve_answers(recognized_text)
    print(answer)
else: 
    print("No voice to query")
