# chroma_server.py
from typing import Annotated
from mcp.server.fastmcp import FastMCP
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 벡터스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
docsearch = Chroma(
    persist_directory="./data/test_db",
    embedding_function=embeddings
)

# ── 메타데이터 로드 (company·fiscal은 metadatas 필드 안에!) ──
client     = docsearch._client
collection = client.get_collection(docsearch._collection_name)

# 여기서 include=["metadatas"] 로 메타데이터 전체를 가져옵니다.
entries = collection.get(include=["metadatas"], limit=100000)

_valid_map     = {}
_valid_tickers = set()
for meta in entries["metadatas"]:
    comp = meta.get("company")
    fy   = meta.get("fiscal")
    if comp is None or fy is None:
        continue
    _valid_map.setdefault(comp, set()).add(fy)
    _valid_tickers.add(comp)
# ─────────────────────────────────────────────────────────

mcp = FastMCP("Chroma")

@mcp.tool()
def retrieve_factual_data(
    question: Annotated[str, "question"],
    ticker:   Annotated[str, "ticker (ex: 'AAPL')"],
    fy:       Annotated[int, "fiscal year (ex: 2020)"]
) -> str:
    # 티커 검증
    if ticker not in _valid_tickers:
        sample = sorted(_valid_tickers)[:10]
        return (
            f"Error: No such ticker '{ticker}'.\n"
            f"Examples of available tickers: {sample}…"
        )

    # 연도 검증
    valid_years = _valid_map[ticker]
    if fy not in valid_years:
        nearby = [y for y in (fy-2, fy-1, fy+1, fy+2) if y in valid_years]
        if nearby:
            return (
                f"Error: No data found for '{ticker}' in fiscal year {fy}.\n"
                f"Available alternative years: {nearby}"
            )
        else:
            return (
                f"Error: No data found for '{ticker}' in fiscal year {fy}"
            )

    # 실제 검색
    retriever = docsearch.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "$and": [
                    {"company": {"$eq": ticker}},
                    {"fiscal":  {"$eq": fy}}
                ]
            }
        }
    )
    results = retriever.invoke(question)
    if results:
        return "\n\n---\n\n".join(doc.page_content for doc in results)
    return "No data returned. Try again with correct ticker and fiscal year."

if __name__ == "__main__":
    mcp.run(transport="stdio")
