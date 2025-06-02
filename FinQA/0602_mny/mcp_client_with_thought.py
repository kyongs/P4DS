import asyncio
import json
import os
from dotenv import load_dotenv, find_dotenv
import sys
import io
import re
from tqdm import tqdm

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_mcp_adapters.client import MultiServerMCPClient


_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

### Query Rewriting
from datetime import datetime
import re

def convert_relative_years_to_absolute(question: str, current_year: int = None) -> str:

    years_in_question = re.findall(r"\b(20\d{2})\b", question)
    if years_in_question:
        reference_year = max(map(int, years_in_question))
    elif current_year is not None:
        reference_year = current_year
    else:
        reference_year = datetime.now().year

    pattern = r"(\d+)\s+years ago"
    matches = re.findall(pattern, question, flags=re.IGNORECASE)

    for match in matches:
        years_ago = int(match)
        absolute_year = reference_year - years_ago
        question = re.sub(
            rf"{match}\s+years ago",
            f"{absolute_year}",
            question,
            flags=re.IGNORECASE
        )

    return question


def wrap_tool_async(tool):
    async def wrapped(input_str: str):
        try:
            data = json.loads(input_str)
            return await tool.ainvoke(data)
        except Exception as e:
            return f"Invalid input: {e}"
    return Tool(
        name=tool.name,
        func=wrapped,
        coroutine=wrapped,
        description=f"{tool.description} (Use with JSON input string)"
    )

def clean_trace(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

results_json = []

async def async_func():
    async with MultiServerMCPClient(
        {
            "chroma": {"command": "python", "args": ["./servers/chroma_server.py"], "transport": "stdio"},
            "fin":    {"command": "python", "args": ["./servers/fin_server.py"],    "transport": "stdio"},
            "math":   {"command": "python", "args": ["./servers/math_server.py"],   "transport": "stdio"},
            "sqlite": {"command": "python", "args": ["./servers/sqlite_server.py"], "transport": "stdio"},
        }
    ) as client:
        raw_tools = client.get_tools()
        tools = [wrap_tool_async(t) for t in raw_tools]

        agent = initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        with open('./data/qa_dict.json', 'r') as f:
            qa_dict = json.load(f)

        for q in tqdm(qa_dict, desc="Processing questions"):
            question = q["Question"]
            question = convert_relative_years_to_absolute(question, current_year=2025)
            # print(question)

            buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = buffer

            try:
                print(f"\n\n▶ 질문: {question}")
                inputs = question if isinstance(question, str) else json.dumps(q)
                output = await agent.ainvoke(inputs, config={"return_messages": True})
                messages = output.get("messages", [])
                final_answer = messages[-1].content if messages else None
            except Exception as e:
                print(f"[ERROR] {e}")
                final_answer = f"[ERROR] {str(e)}"
            finally:
                sys.stdout = original_stdout

            trace_log = buffer.getvalue()
            trace_log = clean_trace(trace_log)
            buffer.close()

            final_answer_match = re.findall(r"Final Answer:\s*(.*)", trace_log)
            final_answer = final_answer_match[-1].strip() if final_answer_match else None

            results_json.append({
                "question": question,
                "final_answer": final_answer,
                "trace": trace_log
            })
    
    with open("./data/v0530/results_thoughts_v9.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

asyncio.run(async_func())
