# mcp_client_with_thought.py

import asyncio
import argparse
import io
import json
import multiprocessing as mp
import os
import re
import sys
from datetime import datetime
from functools import partial

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_mcp_adapters.client import MultiServerMCPClient

# ─────────────────────────────────────────────────────────
# 1) load environment variables & check API key
# ─────────────────────────────────────────────────────────
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

# ─────────────────────────────────────────────────────────
# 2) helper: parse command-line arguments
# ─────────────────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(description="FinQA MCP Client with Thought Processing")
    parser.add_argument(
        "--input", "-i",
        default="./data/qa_dict.json",
        help="Path to input JSON file containing questions (default: ./data/qa_dict.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results/v0602/results_thoughts_v2.json",
        help="Path to output JSON file for results (default: ./results/v0602/results_thoughts_v2.json)"
    )
    parser.add_argument(
        "--cpu-usage", "-c",
        type=float, default=0.75,
        help="CPU usage percentage (0.0-1.0, default: 0.75)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float, default=0,
        help="Model temperature (default: 0)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# 3) helper: convert “X years ago” → absolute year
# ─────────────────────────────────────────────────────────
def convert_relative_years_to_absolute(question: str, current_year: int = None) -> str:
    """Convert expressions like '2 years ago' to an absolute year."""
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


# ─────────────────────────────────────────────────────────
# 4) helper: wrap an MCP tool for LangChain compatibility
# ─────────────────────────────────────────────────────────
def wrap_tool_async(tool):
    """
    Given an MCP tool object, return a LangChain Tool wrapper that expects
    a JSON string as input, parses it, invokes `tool.ainvoke(data)`, and returns
    its result.
    """
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
        description=f"{tool.description} (Use with a JSON input string)"
    )


# ─────────────────────────────────────────────────────────
# 5) helper: clean trace logs (remove ANSI codes & extra blank lines)
# ─────────────────────────────────────────────────────────
def clean_trace(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────
# 6) Core processing: process a single question (with decomposition)
# ─────────────────────────────────────────────────────────
async def process_single_question(question_data, model_name, temperature, server_configs):
    """
    1) Decompose the original question into subtasks via the `decompose_query` tool.
    2) For each subtask:
       - Run a Zero-Shot ReAct agent (with all available tools) on that subtask
       - Capture both the agent’s answer and its printed trace
    3) Concatenate all sub-answers into one final_answer string
    4) Concatenate all sub-traces into one combined trace
    5) Return (index, { "question": original, "final_answer": ..., "trace": ... })
    """
    index, question_dict = question_data
    original_question = question_dict["Question"]

    # (a) Convert any relative‐year phrases before decomposition
    question = convert_relative_years_to_absolute(original_question, current_year=2025)

    # (b) Instantiate the ChatOpenAI LLM for all agent calls
    model = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)

    # (c) Open an MCP client that will launch all servers (chroma, fin, math, sqlite, decomposition, …)
    async with MultiServerMCPClient(server_configs) as client:
        raw_tools = client.get_tools()
        # Wrap each raw MCP tool into a LangChain Tool
        tools_for_agent = [wrap_tool_async(t) for t in raw_tools]

        # (d) Find the raw decomposition tool (`decompose_query`) among raw_tools
        decomp_tool = None
        for t in raw_tools:
            if t.name.lower() == "decompose_query":
                decomp_tool = t
                break

        # (e) Call the decomposition tool to get a list of subtasks
        if decomp_tool is None:
            subtasks = [question]
        else:
            try:
                # The tool expects a JSON‐like dict with key “query”
                decomp_result = await decomp_tool.ainvoke({"query": question})
                if isinstance(decomp_result, list):
                    subtasks = decomp_result
                elif isinstance(decomp_result, str):
                    # Maybe it returned a JSON‐encoded string
                    try:
                        parsed = json.loads(decomp_result)
                        if isinstance(parsed, list):
                            subtasks = parsed
                        else:
                            subtasks = [question]
                    except:
                        subtasks = [question]
                else:
                    subtasks = [question]
            except Exception:
                subtasks = [question]

        # (f) Define a single LangChain agent bound to all wrapped tools
        agent = initialize_agent(
            tools=tools_for_agent,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        all_subanswers = []
        combined_trace = []

        # (g) Loop over each subtask, run the agent, capture answer + trace
        for (step_idx, subq) in enumerate(subtasks):
            buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buffer  # redirect prints into buffer

            try:
                # Print a header so the buffer shows which subtask is running
                print(f"\n\n▶ Subtask {step_idx+1}/{len(subtasks)}: {subq}")
                # Invoke the agent on the subtask
                agent_output = await agent.ainvoke(subq, config={"return_messages": True})
                messages = agent_output.get("messages", [])
                sub_answer = messages[-1].content if messages else ""
            except Exception as e:
                sub_answer = f"[ERROR] {e}"
            finally:
                sys.stdout = old_stdout
                trace_output = buffer.getvalue()
                buffer.close()

            all_subanswers.append(sub_answer)
            combined_trace.append(trace_output)

        # (h) Combine all sub-answers into one final answer string.
        #     You can customize formatting here. For example, add bullet points, etc.
        #     In this version, we simply number them and separate by two newlines.
        final_answer = "\n\n".join(
            f"Subtask {i+1} answer: {ans}"
            for i, ans in enumerate(all_subanswers)
        )

        # (i) Combine all traces into one big trace string, separated by a visible separator
        trace_log = "\n\n--- Subtask Trace Separator ---\n\n".join(combined_trace)

        # (j) Return the result dictionary
        return index, {
            "question": original_question,
            "final_answer": final_answer,
            "trace": trace_log
        }


def process_question_sync(question_data, model_name, temperature, server_configs):
    """
    Synchronous wrapper so we can use multiprocessing. This simply
    runs the async function in an event loop and returns its result.
    """
    return asyncio.run(
        process_single_question(question_data, model_name, temperature, server_configs)
    )


# ─────────────────────────────────────────────────────────
# 7) Main async function: load questions, spawn processes, save results
# ─────────────────────────────────────────────────────────
async def async_func():
    args = parse_arguments()

    # Determine # of worker processes
    num_processes = max(1, int(mp.cpu_count() * args.cpu_usage))
    print(f"Using {num_processes} processes (CPU usage: {args.cpu_usage * 100:.1f}%)")

    # Define server configurations (including the new decomposition server)
    server_configs = {
        "chroma": {
            "command": "python",
            "args": ["./servers/chroma_server.py"],
            "transport": "stdio"
        },
        "fin": {
            "command": "python",
            "args": ["./servers/fin_server.py"],
            "transport": "stdio"
        },
        "math": {
            "command": "python",
            "args": ["./servers/math_server.py"],
            "transport": "stdio"
        },
        "sqlite": {
            "command": "python",
            "args": ["./servers/sqlite_server.py"],
            "transport": "stdio"
        },
        # New decomposition server entry:
        "decompose": {
            "command": "python",
            "args": ["./servers/decomposition_server.py"],
            "transport": "stdio"
        }
    }

    # Load the list of questions from JSON
    with open(args.input, "r") as f:
        qa_dict = json.load(f)

    print(f"Processing {len(qa_dict)} questions from {args.input}")

    # Create (index, question_dict) pairs so results can be reassembled in order
    indexed_questions = [(i, q) for i, q in enumerate(qa_dict)]

    # Build the partial function that each worker will call
    process_func = partial(
        process_question_sync,
        model_name=args.model,
        temperature=args.temperature,
        server_configs=server_configs
    )

    results_dict = {}
    with mp.Pool(processes=num_processes) as pool:
        for index, result in tqdm(
            pool.imap(process_func, indexed_questions),
            total=len(indexed_questions),
            desc="Processing questions"
        ):
            results_dict[index] = result

    # Reassemble results in original order
    results_json = [results_dict[i] for i in range(len(qa_dict))]

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save final JSON
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")


# ─────────────────────────────────────────────────────────
# 8) Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run the main async function
    asyncio.run(async_func())
