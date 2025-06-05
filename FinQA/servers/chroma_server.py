# chroma_server.py
from typing import Annotated, List
from mcp.server.fastmcp import FastMCP

# Use the new OpenAI client import
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os
import json
import logging

# ─────────────────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ─────────────────────────────────────────────────────────
# 환경변수 로드 및 OpenAI 초기화
# ─────────────────────────────────────────────────────────
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

# Instantiate the new client (reads API key from env automatically)
client = OpenAI()

# ─────────────────────────────────────────────────────────
# 벡터스토어 초기화
# ─────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
docsearch_long = Chroma(
    persist_directory="./data/test_db_1000",
    embedding_function=embeddings,
    collection_name="finqa",
)
docsearch_small = Chroma(
    persist_directory="./data/test_db_600",
    embedding_function=embeddings,
    collection_name="finqa",
)

# ─────────────────────────────────────────────────────────
# 메타데이터 로드: company·fiscal 필드 추출
# ─────────────────────────────────────────────────────────
client_chroma_long     = docsearch_long._client
collection_meta_long   = client_chroma_long.get_collection(docsearch_long._collection_name)


_valid_map     = {}   # {company: set(fiscal_year)}
_valid_tickers = set()
entries = collection_meta_long.get(include=["metadatas"], limit=100000)
for meta in entries["metadatas"]:
    comp = meta.get("company")
    fy   = meta.get("fiscal")
    if comp is None or fy is None:
        continue
    _valid_map.setdefault(comp, set()).add(fy)
    _valid_tickers.add(comp)

def extract_keywords_using_llm(question: str) -> List[str]:
    """
    OpenAI LLM을 사용하여 입력된 question에서 핵심 키워드를 추출하여 리스트로 반환합니다.
    반환 형식: ['keyword1', 'keyword2', ...]
    """
    prompt = (
        "Extract the most important financial keywords from the following question. "
        "Each keyword should include the company name or ticker as a prefix. "
        "Answer with a JSON array of lowercase strings, and do NOT include any additional text or explanation. "
        "For example: [\"aapl_net_sales\", \"blackrock_operating_income\"]\n\n"
        f"QUESTION: \"{question}\"\n"
        "OUTPUT (JSON array):"
    )


    try:
        # ──────────────────────────────────────────────────────
        #   Use the new client: client.chat.completions.create(...)
        # ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Return only a plain JSON array of keywords."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=64,
        )
        # The new response object gives access via attributes:
        content = response.choices[0].message.content.strip()
        # logging.info(f"QUESTION: {question} ////// OUTPUT: {content}")

        parsed = json.loads(content)
        if isinstance(parsed, list):
            keywords = [kw.strip().lower() for kw in parsed if isinstance(kw, str) and kw.strip()]
        else:
            keywords = []
    except Exception as e:
        logging.warning(f"extract_keywords_using_llm: LLM call or JSON parse failed ({e})")
        keywords = []

    return keywords

# ─────────────────────────────────────────────────────────
# FastMCP 도구 정의
# ─────────────────────────────────────────────────────────
mcp = FastMCP("Chroma")

@mcp.tool(description="Retrieve financial facts. Required arguments: question (str), ticker (str), fy (int).")
def retrieve_factual_data(
    question: Annotated[str, "question"],
    ticker:   Annotated[str, "ticker (ex: 'AAPL')"],
    fy:       Annotated[int, "fiscal year (ex: 2020)"]
) -> str:
    """
    1) 질문을 기반으로 유사도 검색(k=5)
    2) OpenAI LLM으로 키워드 추출 (extract_keywords_using_llm)
       - 추출된 키워드를 ticker_prefix 형태로 변환
    3) 각 키워드별 검색(k=10)
    4) 중복 제거 후 결과 반환
    """

    # ── 티커 검증
    if ticker not in _valid_tickers:
        sample = sorted(_valid_tickers)[:10]
        return (
            f"Error: No such ticker '{ticker}'.\n"
            f"Examples of available tickers: {sample}…"
        )

    # ── 연도 검증
    valid_years = _valid_map[ticker]
    if fy not in valid_years:
        nearby = [y for y in (fy-2, fy-1, fy+1, fy+2) if y in valid_years]
        if nearby:
            return (
                f"Error: No data found for '{ticker}' in fiscal year {fy}.\n"
                f"Available alternative years: {nearby}"
            )
        else:
            return f"Error: No data found for '{ticker}' in fiscal year {fy}"

    # ── 메타데이터 필터: company=f'{ticker}', fiscal={fy}
    filter_cond = {
        "$and": [
            {"company": {"$eq": ticker}},
            {"fiscal":  {"$eq": fy}}
        ]
    }

    # ── 최종 반환할 결과를 모을 구조
    retrieved_ids   = set()
    retrieved_texts = []

    # ─────────────────────────────────────────────────────────
    # (1) 전체 질문으로 유사도 검색 (k=5)
    # ─────────────────────────────────────────────────────────
    try:
        retriever_q = docsearch_long.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": filter_cond
            }
        )
        results_q = retriever_q.invoke(question)
        for doc in results_q:
            doc_id = getattr(doc, "id", None)
            if doc_id and doc_id not in retrieved_ids:
                retrieved_ids.add(doc_id)
                retrieved_texts.append(doc.page_content)
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────
    # (2) LLM으로 키워드 추출 (extract_keywords_using_llm 함수 사용)
    # ─────────────────────────────────────────────────────────
    keywords = extract_keywords_using_llm(question)
    # logging.info(f"[extract_keywords_using_llm] question: \"{question}\" -> keywords: {keywords}")

    # ─────────────────────────────────────────────────────────
    # (3) 키워드별 검색 (각 키워드마다 k=3)
    # ─────────────────────────────────────────────────────────
    for kw in keywords:
        try:
            retriever_kw = docsearch_small.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": filter_cond
                }
            )
            results_kw = retriever_kw.invoke(kw)
            for doc in results_kw:
                doc_id = getattr(doc, "id", None)
                if doc_id and doc_id not in retrieved_ids:
                    retrieved_ids.add(doc_id)
                    retrieved_texts.append(doc.page_content)
        except Exception:
            continue

    # ─────────────────────────────────────────────────────────
    # (4) 최종 결과 반환: "문서1\n\n---\n\n문서2\n\n---\n\n..."
    # ─────────────────────────────────────────────────────────
    if retrieved_texts:
        combined_text = "\n\n---\n\n".join(retrieved_texts)
        return combined_text
    else:
        return "No data returned. Try again with correct ticker and fiscal year."

if __name__ == "__main__":
    mcp.run(transport="stdio")
