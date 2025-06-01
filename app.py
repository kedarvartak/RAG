from flask import Flask, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)


GOOGLE_API_KEY_HARDCODED = ""


documents = []
text_splits = []
embeddings_model = None
vectorstore = None
retriever_vectordb = None
keyword_retriever = None
ensemble_retriever = None
llm = None 
rag_chain = None 

def initialize_rag():
    global documents, text_splits, embeddings_model, vectorstore
    global retriever_vectordb, keyword_retriever, ensemble_retriever, llm, rag_chain

    loader = TextLoader("./data.txt", encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    raw_text_splits = text_splitter.split_documents(documents)
    
    text_splits = [split for split in raw_text_splits if split.page_content and split.page_content.strip()]

    embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(text_splits, embeddings_model)
    
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 3 

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever],
        weights=[0.5, 0.5]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY_HARDCODED,
        temperature=0.5,
        max_output_tokens=512
    )

    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG pipeline with Gemini LLM initialization attempted (using hardcoded API key).")

initialize_rag()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/rag', methods=['POST'])
def rag_query():
    data = request.get_json()
    query = data.get('query')
    
    response = rag_chain.invoke(query)
    
    return jsonify({"query": query, "answer": response})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True) 