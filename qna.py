from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator


load_dotenv()
llm_model = "gpt-3.5-turbo-0301"
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

docs = loader.load()
print(docs[0])


embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

docs = db.similarity_search(query)
print(len(docs))
print(docs[0])

retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

display(Markdown(response))

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
response = index.query(query, llm=llm)

print(response)




