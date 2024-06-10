from langchain_community.document_loaders import YoutubeLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)
    return db, docs

# db,docs = create_vector_db_from_yt('https://www.youtube.com/watch?v=lG7Uxts9SXs')


def get_response_from_query(db, query, k):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

    prompt = PromptTemplate(
        input_variables = ["question","docs"],
        template=(
        "You are an expert assistant who answers questions based on provided youtube transcript. "
        "Here is a transcript of a YouTube video:\n\n"
        "{docs}\n\n"
        "Based on the transcript, please answer the following question in detail:\n\n"
        "Question: {question}\n\n"
        "Answer:"
        )
    )
    chain = LLMChain(llm=llm,prompt=prompt)

    response = chain.run(question=query,docs = docs_page_content)
    response = response.replace("\n","")
    return response

# db,docs = create_vector_db_from_yt('https://www.youtube.com/watch?v=lG7Uxts9SXs')
dbp, docsp = create_vector_db_from_pdf('/Users/vanshikasharma/Documents/GitHub/LLM_Basic_and_Advanced_Creations/Vanshika_Detailed_CV__Copy___Copy_-2.pdf')
query = 'resume summary?'
k=4
print(get_response_from_query(dbp, query, k))


