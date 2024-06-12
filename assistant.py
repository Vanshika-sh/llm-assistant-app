import requests
from langchain_community.document_loaders import YoutubeLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = OpenAIEmbeddings()

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def create_vector_db_from_pdf(pdf_path: str) -> FAISS:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs

def create_vector_db_from_yt(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs

def create_vector_db_from_url(url: str) -> FAISS:
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    text_content = response.text
    document = Document(page_content=text_content)
    documents = [document]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    return db, docs

def get_response_from_query(db, query, k):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model='gpt-3.5-turbo-instruct')

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=(
            "You are an expert assistant who answers questions based on provided youtube transcript. "
            "Here is a transcript of a YouTube video:\n\n"
            "{docs}\n\n"
            "Based on the transcript, please answer the following question in detail:\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

# Example usage for a PDF
dbp, docsp = create_vector_db_from_pdf('/Users/vanshikasharma/Documents/GitHub/LLM_Basic_and_Advanced_Creations/Vanshika_Resume_May24.pdf')
query = 'resume summary?\n\n'
k = 4
print(get_response_from_query(dbp, query, k))

# Example usage for a YouTube video
db_yt, docs_yt = create_vector_db_from_yt('https://www.youtube.com/watch?v=lG7Uxts9SXs')
query = 'What is the main topic of the video?\n\n'
k = 4
print(get_response_from_query(db_yt, query, k))

# Example usage for a URL
db_url, docs_url = create_vector_db_from_url('https://theaipursuit1.substack.com/p/unlocking-the-power-of-knowledge')
query = 'What is the article about?'
k = 4
print(get_response_from_query(db_url, query, k))







