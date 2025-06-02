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

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def parse_arguments():
    parser = argparse.ArgumentParser(description="FinQA MCP Client with Thought Processing")
    parser.add_argument("--input", "-i", 
                       default="./data/qa_dict.json",
                       help="Path to input JSON file containing questions (default: ./data/qa_dict.json)")
    parser.add_argument("--output", "-o",
                       default="./results/v0602/results_thoughts_v2.json", 
                       help="Path to output JSON file for results (default: ./results/v0602/results_thoughts_v1.json)")
    parser.add_argument("--cpu-usage", "-c",
                       type=float, default=0.75,
                       help="CPU usage percentage (0.0-1.0, default: 0.75)")
    parser.add_argument("--model", "-m",
                       default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", "-t",
                       type=float, default=0,
                       help="Model temperature (default: 0)")
    return parser.parse_args()


def convert_relative_years_to_absolute(question: str, current_year: int = None) -> str:
    """Convert relative year expressions like '2 years ago' to absolute years"""
    # Find years already mentioned in the question
    years_in_question = re.findall(r"\b(20\d{2})\b", question)
    if years_in_question:
        reference_year = max(map(int, years_in_question))
    elif current_year is not None:
        reference_year = current_year
    else:
        reference_year = datetime.now().year

    # Find and replace "X years ago" patterns
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
    """Wrap MCP tools for LangChain agent compatibility"""
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
    """Clean trace logs by removing ANSI codes and extra newlines"""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def process_single_question(question_data, model_name, temperature, server_configs):
    """Process a single question with its index to maintain order"""
    index, question_dict = question_data
    question = question_dict["Question"]
    question = convert_relative_years_to_absolute(question, current_year=2025)
    
    model = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)
    
    async with MultiServerMCPClient(server_configs) as client:
        raw_tools = client.get_tools()
        tools = [wrap_tool_async(t) for t in raw_tools]

        agent = initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        # Capture stdout for trace logging
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer

        try:
            print(f"\n\n▶ 질문 {index}: {question}")
            inputs = question if isinstance(question, str) else json.dumps(question_dict)
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

        # Extract final answer from trace if needed
        final_answer_match = re.findall(r"Final Answer:\s*(.*)", trace_log)
        final_answer = final_answer_match[-1].strip() if final_answer_match else None

        return index, {
            "question": question,
            "final_answer": final_answer,
            "trace": trace_log
        }


def process_question_sync(question_data, model_name, temperature, server_configs):
    """Synchronous wrapper for multiprocessing"""
    return asyncio.run(process_single_question(question_data, model_name, temperature, server_configs))


async def async_func():
    args = parse_arguments()
    
    # Calculate number of processes
    num_processes = max(1, int(mp.cpu_count() * args.cpu_usage))
    print(f"Using {num_processes} processes (CPU usage: {args.cpu_usage*100:.1f}%)")
    
    # Server configurations
    server_configs = {
        "chroma": {"command": "python", "args": ["./servers/chroma_server.py"], "transport": "stdio"},
        "fin":    {"command": "python", "args": ["./servers/fin_server.py"],    "transport": "stdio"},
        "math":   {"command": "python", "args": ["./servers/math_server.py"],   "transport": "stdio"},
        "sqlite": {"command": "python", "args": ["./servers/sqlite_server.py"], "transport": "stdio"},
    }
    
    # Load questions
    with open(args.input, 'r') as f:
        qa_dict = json.load(f)
    
    print(f"Processing {len(qa_dict)} questions from {args.input}")
    
    # Create question data with indices to maintain order
    indexed_questions = [(i, q) for i, q in enumerate(qa_dict)]
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_question_sync,
        model_name=args.model,
        temperature=args.temperature,
        server_configs=server_configs
    )
    
    # Process questions in parallel
    results_dict = {}
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm for progress tracking
        for index, result in tqdm(
            pool.imap(process_func, indexed_questions),
            total=len(indexed_questions),
            desc="Processing questions"
        ):
            results_dict[index] = result
    
    # Sort results by index to maintain original order
    results_json = [results_dict[i] for i in range(len(qa_dict))]
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(async_func())
