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
# 6) NEW: LLM-based composite answer & validation
# ─────────────────────────────────────────────────────────
async def compose_final_answer(model, all_subanswers, combined_trace):
    """
    LLM에게 subtask answer들과 전체 trace를 주고 최종 답을 합성하게 한다.
    """
    prompt = (
        "아래는 복잡한 금융 질문을 여러 서브태스크로 분해해서 얻은 중간 답변들과 trace입니다.\n"
        "각 subtask의 답변과 reasoning, 계산을 고려해서, 최종 질문에 대한 정확하고 일관된 하나의 Final Answer만 한국어로 작성하세요.\n"
        "서브태스크별 답변들:\n"
        + "\n".join([f"서브태스크 {i+1} 답변: {ans}" for i, ans in enumerate(all_subanswers)])
        + "\n\n전체 Trace Log:\n"
        + combined_trace
    )
    resp = await model.ainvoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)

async def validate_final_answer(model, question, final_answer, combined_trace):
    """
    LLM에게 최종 답변과 trace를 주고 논리적/수치적 일관성 검증을 시킨다.
    """
    prompt = (
        "아래는 복잡한 금융 질문에 대해 단계적으로 reasoning을 수행한 Trace와, 도출된 Final Answer입니다.\n"
        "Final Answer가 trace의 reasoning/계산과 논리적으로 일치하는지, plausible한지 한국어로 설명과 함께 '맞다/틀리다'로 평가하세요.\n"
        "불일치나 의심점이 있다면 구체적으로 지적하고, 보수적으로 판단하세요.\n\n"
        f"원본 질문: {question}\n\n"
        f"Final Answer: {final_answer}\n\n"
        f"Trace Log:\n{combined_trace}\n"
    )
    resp = await model.ainvoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)

# ─────────────────────────────────────────────────────────
# 7) Core processing: process a single question (with decomposition)
# ─────────────────────────────────────────────────────────
async def process_single_question(question_data, model_name, temperature, server_configs):
    index, question_dict = question_data
    original_question = question_dict["Question"]
    # (a) Convert any relative-year phrases before decomposition
    question = convert_relative_years_to_absolute(original_question, current_year=2025)
    # (b) Instantiate the ChatOpenAI LLM for all agent calls
    model = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)

    async with MultiServerMCPClient(server_configs) as client:
        raw_tools = client.get_tools()
        tools_for_agent = [wrap_tool_async(t) for t in raw_tools]

        # (d) Find the decomposition tool (`decompose_query`)
        decomp_tool = None
        for t in raw_tools:
            if t.name.lower() == "decompose_query":
                decomp_tool = t
                break

        # (e) Decompose question into subtasks
        if decomp_tool is None:
            subtasks = [question]
        else:
            try:
                decomp_result = await decomp_tool.ainvoke({"query": question})
                if isinstance(decomp_result, list):
                    subtasks = decomp_result
                elif isinstance(decomp_result, str):
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

        agent = initialize_agent(
            tools=tools_for_agent,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        all_subanswers = []
        combined_trace = []

        for (step_idx, subq) in enumerate(subtasks):
            buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buffer  # redirect prints into buffer

            try:
                print(f"\n\n▶ Subtask {step_idx+1}/{len(subtasks)}: {subq}")
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

        subanswers_section = "\n\n".join(
            f"Subtask {i+1} answer: {ans}"
            for i, ans in enumerate(all_subanswers)
        )
        trace_log = "\n\n--- Subtask Trace Separator ---\n\n".join(combined_trace)
        # (i) Use LLM to synthesize/composite the real final answer
        composite_llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)
        composite_final_answer = await compose_final_answer(composite_llm, all_subanswers, trace_log)
        # (j) Validation step
        validation_result = await validate_final_answer(composite_llm, question, composite_final_answer, trace_log)
        # (k) Return the result dictionary with detailed trace info
        return index, {
            "question": original_question,
            "final_answer": composite_final_answer,
            "subanswers": all_subanswers,
            "trace": {
                "subtask_answers": subanswers_section,
                "trace_log": trace_log,
                "composite_llm_output": composite_final_answer,
                "validation_llm_output": validation_result
            }
        }

def process_question_sync(question_data, model_name, temperature, server_configs):
    return asyncio.run(
        process_single_question(question_data, model_name, temperature, server_configs)
    )

# ─────────────────────────────────────────────────────────
# 8) Main async function: load questions, spawn processes, save results
# ─────────────────────────────────────────────────────────
async def async_func():
    args = parse_arguments()
    num_processes = max(1, int(mp.cpu_count() * args.cpu_usage))
    print(f"Using {num_processes} processes (CPU usage: {args.cpu_usage * 100:.1f}%)")

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
        "decompose": {
            "command": "python",
            "args": ["./servers/decomposition_server.py"],
            "transport": "stdio"
        },
    }

    with open(args.input, "r") as f:
        qa_dict = json.load(f)

    print(f"Processing {len(qa_dict)} questions from {args.input}")
    indexed_questions = [(i, q) for i, q in enumerate(qa_dict)]

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

    results_json = [results_dict[i] for i in range(len(qa_dict))]
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")

# ─────────────────────────────────────────────────────────
# 9) Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(async_func())

