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

_ = load_dotenv(find_dotenv())

with open('./data/qa_dict.json', 'r') as f:
    qa_dict = json.load(f)

results_list = [None] * len(qa_dict)

OPENAI_API_KEY = "apikey"

model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

async def async_func():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./servers/math_server.py"],
                "transport": "stdio",
            },
            "fin": {
                "command": "python",
                "args": ["./servers/fin_server.py"],
                "transport": "stdio",
            },
            "chroma": {
                "command": "python",
                "args": ["./servers/chroma_server.py"],
                "transport": "stdio",
            },
            "sqlite": {
                "command": "python",
                "args": ["./servers/sqlite_server.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        for i, item in enumerate(qa_dict):
            agent = create_react_agent(model, client.get_tools())
            result = await agent.ainvoke({"messages": item['Question']},
            config={"recursion_limit": 50}
            )
            print(result)
            results_list[i] = result['messages'][-1].content
            print(results_list[i])

asyncio.run(async_func())

output_data = []
for i, item in enumerate(qa_dict):
    output_data.append({
        'Question': item['Question'],
        'Output': results_list[i]
    })

with open('./data/results.json', 'w') as f:
    json.dump(output_data, f, indent=4)
