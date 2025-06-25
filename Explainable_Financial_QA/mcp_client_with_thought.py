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
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

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
        description=f"{tool.description} (Use with a JSON input string)"
    )

def clean_trace(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_decomposition_result(result):
    """Parse decomposition result, handling both dict and string formats."""
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        try:
            # Try to parse as JSON string
            if result.strip().startswith('{'):
                return json.loads(result)
            else:
                # Fallback for non-JSON string - treat as simple query
                return {
                    "needs_decomposition": False,
                    "complexity_reason": "String response - defaulting to simple execution",
                    "execution_plan": {
                        "type": "simple",
                        "tasks": [
                            {
                                "id": "task_1",
                                "description": result,
                                "depends_on": [],
                                "execution_group": 1
                            }
                        ]
                    }
                }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "needs_decomposition": False,
                "complexity_reason": "Failed to parse JSON string",
                "execution_plan": {
                    "type": "simple",
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Fallback execution",
                            "depends_on": [],
                            "execution_group": 1
                        }
                    ]
                }
            }
    else:
        # Fallback for unexpected types
        return {
            "needs_decomposition": False,
            "complexity_reason": f"Unexpected result type: {type(result)}",
            "execution_plan": {
                "type": "simple",
                "tasks": [
                    {
                        "id": "task_1",
                        "description": str(result),
                        "depends_on": [],
                        "execution_group": 1
                    }
                ]
            }
        }

async def execute_task_with_context(agent, task_description: str, context_results: Dict[str, str], task_id: str) -> Tuple[str, str]:
    """
    Execute a single task with context from previous tasks.
    
    Args:
        agent: The langchain agent
        task_description: Description of the task to execute
        context_results: Results from previous tasks that this task depends on
        task_id: Unique identifier for this task
    
    Returns:
        Tuple of (task_result, trace_output)
    """
    # Prepare the task with context if available
    if context_results:
        context_info = "Context from previous tasks:\n"
        for dep_id, result in context_results.items():
            context_info += f"- {dep_id}: {result}\n"
        full_task = f"{context_info}\nTask: {task_description}"
    else:
        full_task = task_description
    
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    
    try:
        print(f"\n\n▶ Executing {task_id}: {task_description}")
        if context_results:
            print(f"  Using context from: {list(context_results.keys())}")
        
        agent_output = await agent.ainvoke(full_task, config={"return_messages": True})
        
        # Try to get answer from output field first
        if isinstance(agent_output, dict) and "output" in agent_output:
            task_result = str(agent_output["output"])
        else:
            # Fallback to messages
            messages = agent_output.get("messages", [])
            task_result = messages[-1].content if messages else ""
        
    except Exception as e:
        task_result = f"[ERROR] {e}"
    finally:
        sys.stdout = old_stdout
        trace_output = buffer.getvalue()
        buffer.close()
        
        # If task_result is still empty, try to parse "Final Answer:" from trace
        if not task_result and trace_output:
            clean_trace_text = clean_trace(trace_output)
            final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|$)", clean_trace_text, re.DOTALL)
            if final_answer_match:
                task_result = final_answer_match.group(1).strip()
    
    return task_result, trace_output

def safe_print(*args, **kwargs):
    """Safe print function that handles closed file errors in multiprocessing"""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (ValueError, OSError, BrokenPipeError):
        # File/pipe is closed or broken, silently ignore
        pass

async def execute_execution_plan(agent, execution_plan: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Execute tasks sequentially based on execution groups.
    No parallel execution within each question to avoid complexity.
    Dependencies are handled by passing context from previous tasks.
    """
    tasks = execution_plan["tasks"]
    execution_type = execution_plan.get("type", "simple")
    
    safe_print(f"Execution plan type: {execution_type}")
    
    # Group tasks by execution group
    groups = {}
    for task in tasks:
        group = task.get("execution_group", 1)
        if group not in groups:
            groups[group] = []
        groups[group].append(task)
    
    safe_print(f"Total execution groups: {len(groups)}")
    
    all_results = []
    all_traces = []
    context_results = {}  # Store results by task_id for dependency resolution
    
    # Execute groups in order - all tasks executed sequentially
    for group_num in sorted(groups.keys()):
        group_tasks = groups[group_num]
        safe_print(f"\n--- Executing Group {group_num} ({len(group_tasks)} tasks) ---")
        
        # Execute all tasks in this group sequentially (no parallel execution)
        for task in group_tasks:
            safe_print(f"  Executing task: {task['description']}")
            
            # Gather context from dependencies
            task_context = {}
            for dep_id in task.get("depends_on", []):
                if dep_id in context_results:
                    task_context[dep_id] = context_results[dep_id]
            
            result, trace = await execute_task_with_context(
                agent, task["description"], task_context, task["id"]
            )
            all_results.append(result)
            all_traces.append(trace)
            context_results[task["id"]] = result
    
    return all_results, all_traces

async def compose_final_answer(model, question, all_subanswers, combined_trace, execution_plan=None):
    """
    Use LLM to synthesize a final answer in English, considering all subtask answers and trace.
    """
    plan_info = ""
    if execution_plan:
        plan_info = f"Execution plan type: {execution_plan.get('type', 'unknown')}\n"
        if execution_plan.get('needs_decomposition', True):
            plan_info += f"Complexity reason: {execution_plan.get('complexity_reason', 'Complex query requiring decomposition')}\n"
        else:
            plan_info += "This was a simple query that didn't require decomposition.\n"
    
    prompt = (
        "Below is the original financial question and intermediate answers and trace logs that have been processed using intelligent decomposition.\n"
        f"{plan_info}"
        "Carefully review all task answers, reasoning, and calculations, then write a single, accurate, and self-consistent final answer in English to the original question.\n"
        "If there are inconsistencies, use the most plausible reasoning and data.\n\n"
        f"Original Question: {question}\n\n"
        "Task answers:\n"
        + "\n".join([f"Task {i+1} answer: {ans}" for i, ans in enumerate(all_subanswers)])
        + "\n\nFull trace log:\n"
        + combined_trace
    )
    resp = await model.ainvoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)

async def validate_final_answer(model, question, final_answer, combined_trace, execution_plan=None):
    """
    Use LLM to validate whether the final answer is logically and numerically consistent with the trace.
    """
    plan_info = ""
    if execution_plan:
        plan_info = f"Execution approach: {execution_plan.get('type', 'unknown')}\n"
    
    prompt = (
        "Below is a trace of step-by-step reasoning for a financial question, as well as a proposed Final Answer.\n"
        f"{plan_info}"
        "Evaluate whether the Final Answer is logically and numerically consistent with the reasoning and calculations in the trace. "
        "Respond in English with a brief explanation, then clearly state 'VALID' if the answer is plausible and matches the reasoning, or 'INVALID' if there are inconsistencies or mistakes. "
        "Be conservative in your judgement and point out any concrete issues you find.\n\n"
        f"Original Question: {question}\n\n"
        f"Final Answer: {final_answer}\n\n"
        f"Trace Log:\n{combined_trace}\n"
    )
    resp = await model.ainvoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp)

async def process_single_question(question_data, model_name, temperature, server_configs):
    index, question_dict = question_data
    original_question = question_dict["Question"]
    question = convert_relative_years_to_absolute(original_question, current_year=2025)
    model = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)

    try:
        async with MultiServerMCPClient(server_configs) as client:
            raw_tools = client.get_tools()
            tools_for_agent = [wrap_tool_async(t) for t in raw_tools]

            # Find decomposition tool
            decomp_tool = None
            for t in raw_tools:
                if t.name.lower() == "decompose_query":
                    decomp_tool = t
                    break

            # Analyze the query for decomposition
            if decomp_tool is None:
                # Fallback to simple execution if no decomposition tool
                execution_plan = {
                    "needs_decomposition": False,
                    "complexity_reason": "No decomposition tool available",
                    "execution_plan": {
                        "type": "simple",
                        "tasks": [
                            {
                                "id": "task_1",
                                "description": question,
                                "depends_on": [],
                                "execution_group": 1
                            }
                        ]
                    }
                }
            else:
                try:
                    decomp_result = await decomp_tool.ainvoke({"query": question})
                    execution_plan = parse_decomposition_result(decomp_result)
                except Exception as e:
                    safe_print(f"[ERROR] Decomposition failed: {e}")
                    execution_plan = {
                        "needs_decomposition": False,
                        "complexity_reason": f"Decomposition error: {e}",
                        "execution_plan": {
                            "type": "simple",
                            "tasks": [
                                {
                                    "id": "task_1",
                                    "description": question,
                                    "depends_on": [],
                                    "execution_group": 1
                                }
                            ]
                        }
                    }

            # Create agent with enhanced system prompt
            system_prompt = """You are a financial data analysis expert. Follow these critical guidelines:

TOOL PRIORITY RULES:
1. ALWAYS use 'retrieve_factual_data' FIRST for any financial question before trying other tools
2. Do NOT use 'list_tables', 'describe_table', or 'read_query' unless retrieve_factual_data fails
3. For financial calculations, use exact company names and ticker symbols

Answer financial questions accurately using the appropriate tools in the correct order."""

            agent = initialize_agent(
                tools=tools_for_agent,
                llm=model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                agent_kwargs={"system_message": system_prompt}
            )

            # Execute the plan
            safe_print(f"\n{'='*60}")
            safe_print(f"Question {index + 1}: {question}")
            safe_print(f"Decomposition needed: {execution_plan.get('needs_decomposition', 'unknown')}")
            safe_print(f"Reason: {execution_plan.get('complexity_reason', 'N/A')}")
            safe_print(f"{'='*60}")
            
            # Check if this is a simple case with one task - use original query directly
            exec_plan = execution_plan["execution_plan"]
            if (exec_plan.get("type") == "simple" and 
                len(exec_plan["tasks"]) == 1 and 
                not execution_plan.get("needs_decomposition", True)):
                
                safe_print("Simple query detected - executing original question directly")
                
                # Execute original question directly without task decomposition
                buffer = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buffer
                
                try:
                    safe_print(f"\n\n▶ Executing original question: {question}")
                    agent_output = await agent.ainvoke(question, config={"return_messages": True})
                    
                    # Try to get answer from output field first
                    if isinstance(agent_output, dict) and "output" in agent_output:
                        direct_result = str(agent_output["output"])
                    else:
                        # Fallback to messages
                        messages = agent_output.get("messages", [])
                        direct_result = messages[-1].content if messages else ""
                        
                except Exception as e:
                    direct_result = f"[ERROR] {e}"
                finally:
                    sys.stdout = old_stdout
                    trace_output = buffer.getvalue()
                    buffer.close()
                    
                    # If direct_result is still empty, try to parse "Final Answer:" from trace
                    if not direct_result and trace_output:
                        clean_trace_text = clean_trace(trace_output)
                        final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|$)", clean_trace_text, re.DOTALL)
                        if final_answer_match:
                            direct_result = final_answer_match.group(1).strip()
                
                all_task_results = [direct_result]
                all_trace_outputs = [trace_output]
                final_answer = direct_result  # Use direct result as final answer for simple cases
                
                # Simple validation for direct execution
                validation_result = f"VALID - Simple query executed directly without decomposition."
                
            else:
                # Complex case - use task decomposition
                safe_print("Complex query detected - using task decomposition")
                all_task_results, all_trace_outputs = await execute_execution_plan(
                    agent, execution_plan["execution_plan"]
                )

                # Prepare results
                subanswers_section = "\n\n".join(
                    f"Task {i+1} answer: {ans}"
                    for i, ans in enumerate(all_task_results)
                )
                trace_log = "\n\n--- Task Trace Separator ---\n\n".join(all_trace_outputs)
                
                # Compose final answer
                composite_llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)
                final_answer = await compose_final_answer(
                    composite_llm, question, all_task_results, trace_log, execution_plan
                )
                validation_result = await validate_final_answer(
                    composite_llm, question, final_answer, trace_log, execution_plan
                )
            
            return index, {
                "question": original_question,
                "final_answer": final_answer,
                "task_results": all_task_results,
                "execution_info": {
                    "needs_decomposition": execution_plan.get("needs_decomposition", False),
                    "complexity_reason": execution_plan.get("complexity_reason", ""),
                    "execution_type": execution_plan["execution_plan"].get("type", "unknown"),
                    "num_tasks": len(execution_plan["execution_plan"]["tasks"]),
                    "task_descriptions": [task["description"] for task in execution_plan["execution_plan"]["tasks"]]
                },
                "trace": {
                    "task_answers": "\n\n".join(f"Task {i+1} answer: {ans}" for i, ans in enumerate(all_task_results)),
                    "trace_log": "\n\n--- Task Trace Separator ---\n\n".join(all_trace_outputs),
                    "composite_llm_output": final_answer,
                    "validation_llm_output": validation_result
                }
            }
    except Exception as e:
        safe_print(f"[ERROR] Error processing question {index + 1}: {e}")
        return index, {
            "question": original_question,
            "final_answer": f"Error: {str(e)}",
            "task_results": [],
            "execution_info": {
                "needs_decomposition": False,
                "complexity_reason": f"Processing error: {e}",
                "execution_type": "error",
                "num_tasks": 0,
                "task_descriptions": []
            },
            "trace": {
                "task_answers": "",
                "trace_log": f"Error occurred: {e}",
                "composite_llm_output": f"Error: {e}",
                "validation_llm_output": "Error during processing"
            }
        }

def process_question_sync(question_data, model_name, temperature, server_configs):
    return asyncio.run(
        process_single_question(question_data, model_name, temperature, server_configs)
    )

async def async_func():
    args = parse_arguments()
    num_processes = max(1, min(50, int(mp.cpu_count() * args.cpu_usage)))
    
    safe_print(f"Using {num_processes} processes (CPU usage: {args.cpu_usage * 100:.1f}%)")

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

    safe_print(f"Processing {len(qa_dict)} questions from {args.input}")
    indexed_questions = [(i, q) for i, q in enumerate(qa_dict)]

    process_func = partial(
        process_question_sync,
        model_name=args.model,
        temperature=args.temperature,
        server_configs=server_configs
    )

    results_dict = {}
    
    with mp.Pool(processes=num_processes) as pool:
        try:
            for index, result in tqdm(
                pool.imap(process_func, indexed_questions),
                total=len(indexed_questions),
                desc="Processing questions"
            ):
                results_dict[index] = result
        except Exception as e:
            safe_print(f"[ERROR] Multiprocessing error: {e}")
            pool.terminate()
            pool.join()
            raise

    results_json = [results_dict[i] for i in range(len(qa_dict))]
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    safe_print(f"Results saved to {args.output}")



if __name__ == "__main__":
    asyncio.run(async_func())

