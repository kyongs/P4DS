# servers/google_search_server.py
import os
import requests
from typing import List, Dict
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID  = os.getenv("GOOGLE_CSE_ID")
if not API_KEY or not CSE_ID:
    raise RuntimeError("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your environment.")

mcp = FastMCP("GoogleSearch")

@mcp.tool(
    annotations={
        "title": "Google Web Search",
        "description": (
            "Use this tool **only when** a question cannot be answered confidently by the local data sources\n"
            "(e.g., your company database or previously retrieved documents). "
            "If the answer is already in the database, or you can derive it from existing facts,\n"
            "do NOT call this tool. Reserve Google Web Search for truly ambiguous queries,\n"
            "real-time updates, or when you need external context that the local system lacks.\n\n"
            "Inputs:\n"
            "  • query (str): The search query string.\n"
            "  • num_results (int): Number of top results to return (1–5).\n\n"
            "Output:\n"
            "  • List[Dict]: A list of result objects, each with keys 'title', 'snippet', and 'link'."
        ),
        "readOnlyHint": True,
        "openWorldHint": True,
        "examples": [
            {"query": "Latest regulatory changes affecting financial reporting", "num_results": 3}
        ]
    }
)
def google_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Returns a list of search results from Google Custom Search.
    """
    if num_results < 1 or num_results > 10:
        raise ValueError("num_results must be between 1 and 5")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": num_results
    }
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    results = []
    for item in items:
        results.append({
            "title":   item.get("title"),
            "snippet": item.get("snippet"),
            "link":    item.get("link")
        })
    return results

if __name__ == "__main__":
    mcp.run(transport="stdio")
