from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
from dotenv import load_dotenv, find_dotenv
import os
from langchain.schema import BaseMessage 
from langchain_core.messages import SystemMessage


_ = load_dotenv(find_dotenv())

with open('./data/qa_dict.json', 'r') as f:
    qa_dict = json.load(f)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

results_json = []          # <-- 새로 쓸 리스트


async def async_func():
    async with MultiServerMCPClient(
        {
            "chroma": {"command": "python", "args": ["./servers/chroma_server.py"], "transport": "stdio"},
            "fin":    {"command": "python", "args": ["./servers/fin_server.py"],    "transport": "stdio"},
            "math":   {"command": "python", "args": ["./servers/math_server.py"],   "transport": "stdio"},
            "sqlite": {"command": "python", "args": ["./servers/sqlite_server.py"], "transport": "stdio"},
            # "google": {"command": "python", "args": ["./servers/google_search_server.py"], "transport": "stdio"},
        }
    ) as client:

        for q in qa_dict:
            agent = create_react_agent(model, client.get_tools())
            # ⭑ invoke 시 intermediate 메시지까지 돌려받게 함
            result = await agent.ainvoke(
                {"messages": f"{q['Question']}.\n You are a financial QA assistant. Think step by step before answering. Use available tools to answer precisely and cite tool calls when needed. "},
                config={"recursion_limit": 30, "return_messages": True}
            )

            # --- 1) 전체 과정(trace) 정리 -------------
            trace = [
                {
                    "role": getattr(m, "role", getattr(m, "type", "assistant")),
                    "content": getattr(m, "content", str(m)),
                }
                for m in result["messages"]               # BaseMessage 객체 → dict
            ]

            # --- 2) Thought 만 골라내기 ---------------
            thoughts = [
                step["content"]
                for step in trace
                if step["role"] == "assistant"
                and step["content"].lstrip().lower().startswith("thought")
            ]

            # --- 3) 결과 리스트에 추가 -----------------
            results_json.append(
                {
                    "question": q["Question"],
                    "output":   result["messages"][-1].content,   # 최종 답
                    "process":  trace,                            # 모든 단계
                    "thoughts": thoughts,                         # Thought 만 모음
                }
            )
# -------------------------------------------------------------------------


# ---- (기존 asyncio.run 이후 부분을 아래로 교체) --------------------------
asyncio.run(async_func())

with open("./data/results.json", "w") as f:
    json.dump(results_json, f, indent=4, ensure_ascii=False)
