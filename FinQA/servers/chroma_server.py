# chroma_server.py
from mcp.server.fastmcp import FastMCP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model='text-embedding-3-small',api_key=OPENAI_API_KEY)

from langchain_chroma import Chroma

docsearch = Chroma(
    persist_directory="./data/test_db",
    embedding_function=embeddings
)

mcp = FastMCP("Chroma")

@mcp.tool()
def retrieve_factual_data(question:str, ticker: str, fy: int) -> str:
  """Search vector DB for the financial reports with the question and ticker symbol and fiscal year. It contains historical data for the company.

   Args:
        question: Question need to be answered
        ticker: Ticker symbol of the company for filtering the documents
        fy: Fiscal year for filtering the documents

    Returns:
        A related document for the question.
  """
  retriever = docsearch.as_retriever(search_kwargs={'k': 1, 'filter':
  {
      "$and": [
          {
              "company": {
                  "$eq": ticker
              }
          },
          {
              "fiscal": {
                  "$eq": fy
              }
          }
      ]
  }})
  result = retriever.invoke(question)
  if result:
    return result[0].page_content
  else:
    return "No data returned. Try again with correct ticker and fiscal year, or different question"
  

if __name__ == "__main__":
    mcp.run()
