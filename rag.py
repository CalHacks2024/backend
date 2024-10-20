from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.docstore.document import Document
import os
import shutil
from consts import CHROMA_PATH
from datasets import load_dataset
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def generate_data_store():
    documents = load_documents()
    save_to_chroma(documents)

def load_documents():
    ds = load_dataset("shefali2023/webmd-data", split="train")
    
    # Correctly wrap the dataset entries in Document objects
    documents = [Document(page_content=doc) for doc in ds['Prompt']]
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)

         # Initialize OpenAI embedding model
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        
        collection = client.create_collection(name="webmd_rag", embedding_function=openai_ef)
        texts = [chunk.page_content for chunk in chunks]
        print("Adding Documents to ChromaDB...")
        # Add texts and their embeddings to Chroma
        collection.add(documents=texts, ids=[str(num) for num in range(len(texts))])
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(e)


def getDB():
    print("Loading persisted ChromaDB...")
    

def generate_questions(client, input_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": QUESTION_PROMPT + " Input Text: " + input_text},
        ]
    )

    return response.choices[0].message.content