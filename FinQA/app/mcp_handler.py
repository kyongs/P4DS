import asyncio
import os
import json
import re
import io
import sys
from typing import Dict, List, Any, Tuple, Callable, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents import Tool, initialize_agent, AgentType

# Load environment variables
_ = load_dotenv(find_dotenv())

class MCPHandler:
    """Handler for MCP client operations and query processing."""
    
    def __init__(self):
        self.client = None
        self.agent = None
        self.tool_count = 0
        self.tool_names = []
        self.tools_info = []
        self.server_tools = {}
        self.recursion_limit = 50
        self.initialized = False
        
        # Calculate server paths based on current file location
        current_file = Path(__file__).resolve()
        app_dir = current_file.parent
        finqa_dir = app_dir.parent
        servers_dir = finqa_dir / "servers"
        
        self.server_paths = {
            "chroma": str(servers_dir / "chroma_server.py"),
            "fin": str(servers_dir / "fin_server.py"),
            "math": str(servers_dir / "math_server.py"),
            "sqlite": str(servers_dir / "sqlite_server.py"),
            "decompose": str(servers_dir / "decomposition_server.py"),
        }
        
        # Debug: Print paths for verification
        print(f"App directory: {app_dir}")
        print(f"FinQA directory: {finqa_dir}")
        print(f"Servers directory: {servers_dir}")
        for server, path in self.server_paths.items():
            exists = Path(path).exists()
            print(f"Server {server}: {path} (exists: {exists})")
    
    async def cleanup_client(self):
        """Safely terminates the existing MCP client."""
        if self.client is not None:
            try:
                # Properly close any pending generators
                # Store reference to exit stack for explicit closing
                if hasattr(self.client, "exit_stack"):
                    try:
                        await self.client.exit_stack.aclose()
                        print("Successfully closed client exit stack")
                    except Exception as e:
                        print(f"Error closing exit stack: {e}")
                
                # Close all sessions if they exist
                if hasattr(self.client, "sessions"):
                    for server_name, session in self.client.sessions.items():
                        try:
                            if hasattr(session, "close"):
                                await session.close()
                            print(f"Closed session for {server_name}")
                        except Exception as e:
                            print(f"Error closing session {server_name}: {e}")
                
                # Now try to exit the client properly using a try-finally block
                try:
                    await self.client.__aexit__(None, None, None)
                finally:
                    self.client = None
                    # Force garbage collection to clean up any remaining references
                    import gc
                    gc.collect()
                    print("MCP client successfully cleaned up")
            except Exception as e:
                import traceback
                print(f"Error during MCP client cleanup: {e}")
                print(traceback.format_exc())
                
                # As a last resort, just set to None to allow garbage collection
                self.client = None
    
    async def initialize(self):
        """Initialize MCP client and agent."""
        # First safely clean up existing client
        await self.cleanup_client()

        # Initialize MultiServerMCPClient with decomposition server
        self.client = MultiServerMCPClient(
            {
                "chroma": {"command": "python", "args": [self.server_paths["chroma"]], "transport": "stdio"},
                "fin":    {"command": "python", "args": [self.server_paths["fin"]],    "transport": "stdio"},
                "math":   {"command": "python", "args": [self.server_paths["math"]],   "transport": "stdio"},
                "sqlite": {"command": "python", "args": [self.server_paths["sqlite"]], "transport": "stdio"},
                "decompose": {"command": "python", "args": [self.server_paths["decompose"]], "transport": "stdio"},
            }
        )
        
        await self.client.__aenter__()
        tools = self.client.get_tools()
        
        # Process tools information
        self.tool_count = len(tools)
        self.tool_names = []
        self.tools_info = []
        
        # Group tools by server
        self._process_server_tools(tools)
        
        # Initialize OpenAI model
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )
        
        # Create ReAct agent with enhanced system prompt
        system_prompt = """You are a financial data analysis expert. Follow these critical guidelines:

TOOL PRIORITY RULES:
1. ALWAYS use 'retrieve_factual_data' FIRST for any financial question before trying other tools
2. Do NOT use 'list_tables', 'describe_table', or 'read_query' unless retrieve_factual_data fails
3. For financial calculations, use exact company names and ticker symbols

Answer financial questions accurately using the appropriate tools in the correct order."""

        # Use langchain agent for better control
        wrapped_tools = [self.wrap_tool_async(t) for t in tools]
        self.agent = initialize_agent(
            tools=wrapped_tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": system_prompt}
        )
        
        self.initialized = True
        return True
    
    def _process_server_tools(self, tools):
        """Process and categorize tools by server."""
        tool_servers = {}
        
        # Try to get server names directly from the client if possible
        try:
            if hasattr(self.client, "server_name_to_tools"):
                # Direct access to server_name_to_tools mapping
                print("Found server_name_to_tools attribute in client")
                server_tool_mapping = self.client.server_name_to_tools
                
                # Create tool servers map from the client's own mapping
                for server_name, server_tools in server_tool_mapping.items():
                    # Initialize each server group
                    tool_servers[server_name] = []
                    
                    # Process each tool in this server's tools
                    for tool in server_tools:
                        # Store basic info for all tools
                        self.tool_names.append(tool.name)
                        self.tools_info.append(f"Tool: {tool.name} - {tool.description}")
                        
                        # Add to the server group
                        tool_servers[server_name].append({
                            "name": tool.name,
                            "full_name": tool.name,
                            "description": tool.description
                        })
                
                # Debug info
                print(f"Found {len(server_tool_mapping)} servers from client mapping")
                for server, s_tools in server_tool_mapping.items():
                    print(f"  Server '{server}' has {len(s_tools)} tools: {[t.name for t in s_tools]}")
                
                print("Using server_name_to_tools for tool mapping")
            else:
                # Fallback to server name extraction
                raise AttributeError("No server_name_to_tools attribute found")
        except (AttributeError, Exception) as e:
            # Fallback method if direct access fails
            print(f"Falling back to name-based server detection: {str(e)}")
            
            # Server names from the MCP client initialization
            server_names = ["chroma", "fin", "math", "sqlite"]
            print(f"Expected servers: {server_names}")
            
            # Initialize server groups
            for server_name in server_names:
                tool_servers[server_name] = []
            
            # Add unknown category for any unmatched tools
            if "unknown" not in tool_servers:
                tool_servers["unknown"] = []
            
            # Categorize tools by server using name prefixes
            for tool in tools:
                # Add to tool names list
                self.tool_names.append(tool.name)
                
                # Use server prefix in tool name to identify the server
                matched_server = False
                for server_name in server_names:
                    # Check if tool name starts with server prefix (e.g., "fin.")
                    if tool.name.startswith(f"{server_name}."):
                        # Extract the tool name without the server prefix
                        tool_name = tool.name.split(".", 1)[1]
                        # Add to the appropriate server group
                        tool_servers[server_name].append({
                            "name": tool_name,
                            "full_name": tool.name,
                            "description": tool.description
                        })
                        matched_server = True
                        break
                
                # If no server prefix was found, add to unknown
                if not matched_server:
                    tool_servers["unknown"].append({
                        "name": tool.name,
                        "full_name": tool.name,
                        "description": tool.description
                    })
                
                # Store detailed tool info
                self.tools_info.append(f"Tool: {tool.name} - {tool.description}")
        
        # Print debug info about tool categorization
        print(f"Final server grouping results: {[server for server in tool_servers.keys()]}")
        for server, tools_list in tool_servers.items():
            print(f"Server '{server}' has {len(tools_list)} tools: {[t['name'] for t in tools_list]}")
        
        # Store server summary
        self.server_tools = tool_servers
    
    def wrap_tool_async(self, tool):
        """Wrap MCP tool for use with langchain agent."""
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

    def convert_relative_years_to_absolute(self, question: str, current_year: int = None) -> str:
        """Convert relative year references to absolute years."""
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

    def parse_decomposition_result(self, result):
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

    async def execute_task_with_context(self, task_description: str, context_results: Dict[str, str], task_id: str) -> Tuple[str, str]:
        """Execute a single task with context from previous tasks."""
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
            
            agent_output = await self.agent.ainvoke(full_task, config={"return_messages": True})
            
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
                clean_trace_text = self.clean_trace(trace_output)
                final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|$)", clean_trace_text, re.DOTALL)
                if final_answer_match:
                    task_result = final_answer_match.group(1).strip()
        
        return task_result, trace_output

    def clean_trace(self, text: str) -> str:
        """Clean ANSI escape codes and excess newlines from trace."""
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    async def execute_execution_plan(self, execution_plan: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Execute tasks sequentially based on execution groups."""
        tasks = execution_plan["tasks"]
        execution_type = execution_plan.get("type", "simple")
        
        print(f"Execution plan type: {execution_type}")
        
        # Group tasks by execution group
        groups = {}
        for task in tasks:
            group = task.get("execution_group", 1)
            if group not in groups:
                groups[group] = []
            groups[group].append(task)
        
        print(f"Total execution groups: {len(groups)}")
        
        all_results = []
        all_traces = []
        context_results = {}  # Store results by task_id for dependency resolution
        task_info = []  # Store task execution info
        
        # Execute groups in order - all tasks executed sequentially
        for group_num in sorted(groups.keys()):
            group_tasks = groups[group_num]
            print(f"\n--- Executing Group {group_num} ({len(group_tasks)} tasks) ---")
            
            # Execute all tasks in this group sequentially
            for task in group_tasks:
                print(f"  Executing task: {task['description']}")
                
                # Gather context from dependencies
                task_context = {}
                for dep_id in task.get("depends_on", []):
                    if dep_id in context_results:
                        task_context[dep_id] = context_results[dep_id]
                
                result, trace = await self.execute_task_with_context(
                    task["description"], task_context, task["id"]
                )
                all_results.append(result)
                all_traces.append(trace)
                context_results[task["id"]] = result
                
                # Store task info for display
                task_info.append({
                    "id": task["id"],
                    "description": task["description"],
                    "result": result,
                    "dependencies": task.get("depends_on", [])
                })
        
        return all_results, all_traces, {"tasks": task_info, "execution_type": execution_type}

    async def compose_final_answer(self, model, question, all_subanswers, combined_trace, execution_plan=None):
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

    async def validate_final_answer(self, model, question, final_answer, combined_trace, execution_plan=None):
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

    async def _execute_simple_query(self, query: str, callback_handler: Tuple[Callable, List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Execute a simple query without decomposition."""
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        
        try:
            print(f"\n\n▶ Executing simple query: {query}")
            agent_output = await self.agent.ainvoke(query, config={"return_messages": True})
            
            # Try to get answer from output field first
            if isinstance(agent_output, dict) and "output" in agent_output:
                final_answer = str(agent_output["output"])
            else:
                # Fallback to messages
                messages = agent_output.get("messages", [])
                final_answer = messages[-1].content if messages else ""
                
        except Exception as e:
            final_answer = f"[ERROR] {e}"
        finally:
            sys.stdout = old_stdout
            trace_output = buffer.getvalue()
            buffer.close()
            
            # If final_answer is still empty, try to parse "Final Answer:" from trace
            if not final_answer and trace_output:
                clean_trace_text = self.clean_trace(trace_output)
                final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|$)", clean_trace_text, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
        
        # Update callback handler with final answer
        callback_func, accumulated_text = callback_handler
        accumulated_text.append(final_answer)
        
        response = {"final_answer": final_answer}
        process_steps = [
            {"role": "human", "content": query},
            {"role": "assistant", "content": final_answer}
        ]
        
        return response, process_steps

    # Explainability functions
    def parse_trace_observations(self, trace: str) -> List[Tuple[str, str]]:
        """Parse trace to extract observations with citations."""
        obs_pattern = re.compile(r"^([\w/.-]+\.pdf-\d+)\s*:\s*(.*)$")
        return [(m.group(1), m.group(2).strip()) for line in trace.splitlines() if (m := obs_pattern.match(line.strip()))]

    def extract_cited_paragraphs(self, trace: str, cited_sources: List[str]) -> str:
        """Extract paragraphs from trace for cited sources."""
        if isinstance(trace, dict):
            trace_str = trace.get("trace_log", "")
        else:
            trace_str = trace
        
        lines = trace_str.splitlines()
        paragraph_blocks = {}
        current_citation = None
        buffer = []
        obs_pattern = re.compile(r"^([\w/.-]+\.pdf-\d+)\s*:\s*(.*)$")

        for line in lines:
            match = obs_pattern.match(line.strip())
            if match:
                if current_citation and current_citation in cited_sources:
                    paragraph_blocks.setdefault(current_citation, []).extend(buffer)
                current_citation = match.group(1).strip()
                buffer = [match.group(2).strip()]
            elif current_citation:
                buffer.append(line.strip())

        if current_citation and current_citation in cited_sources:
            paragraph_blocks.setdefault(current_citation, []).extend(buffer)

        return "\n---\n".join([
            f"{src}:\n" + "\n".join(lines).strip()
            for src, lines in paragraph_blocks.items()
        ])

    def build_explanation_prompt(self, question: str, final_answer: str, trace: str) -> str:
        """Build prompt for explainability analysis."""
        trace_data = trace
        if isinstance(trace_data, dict):
            trace_str = trace_data.get("trace_log", "")
        else:
            trace_str = trace_data
        
        obs_block = "\n".join(f"{src}: {para}" for src, para in self.parse_trace_observations(trace_str))
        return (
            f"### Question\n{question}\n\n"
            f"### Final Answer (agent)\n{final_answer}\n\n"
            f"### Agent Trace\n{trace_str}\n\n"
            f"### Parsed Observations\n{obs_block}"
        )

    async def generate_explanation(self, question: str, final_answer: str, trace: str) -> Dict[str, str]:
        """Generate explanation for the agent's reasoning process."""
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        system_prompt = (
            "You are an expert financial QA explainer. Given a ReAct trace, explain "
            "step by step *why* the agent produced its final answer. Quote and cite the "
            "exact document lines (file path + page) that justify each step, then end "
            "with a one-sentence summary restating the final answer. Use clear, "
            "reader-friendly English and bullet points.\n\n"
            "At the end, include a separate section titled 'Citation:' listing all cited sources "
            "in the format: <file name - page>."
        )
        
        user_prompt = self.build_explanation_prompt(question, final_answer, trace)
        
        try:
            resp = await model.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            full_output = resp.content.strip()

            citation_match = re.search(r"Citation:\s*(.*)$", full_output, re.DOTALL | re.IGNORECASE)
            citation = citation_match.group(1).strip() if citation_match else ""
            explanation = full_output.replace(citation_match.group(0), "").strip() if citation_match else full_output

            # Parse cited sources
            cited_sources = re.findall(r"([\w/.-]+\.pdf-\d+)", citation)
            
            # Extract citation paragraphs
            trace_for_extraction = trace
            if isinstance(trace, dict):
                trace_for_extraction = trace.get("trace_log", "")
            
            citation_paragraph = self.extract_cited_paragraphs(trace_for_extraction, cited_sources)

            return {
                "explanation": explanation,
                "citation": citation,
                "citation_paragraph": citation_paragraph
            }
        except Exception as e:
            return {
                "explanation": f"[ERROR] Failed to generate explanation: {e}",
                "citation": "",
                "citation_paragraph": ""
            }
    
    async def process_query(self, 
                           query: str, 
                           callback_handler: Optional[Tuple[Callable, List[str]]] = None,
                           thread_id: str = None,
                           timeout_seconds: int = 60) -> Tuple[Dict[str, Any], str, List[Dict[str, str]]]:
        """
        Process a user query with intelligent decomposition and task execution.
        
        Args:
            query: The user's query text
            callback_handler: Tuple of (callback_function, accumulated_text_list) for streaming
            thread_id: Unique ID for this conversation thread
            timeout_seconds: Maximum processing time in seconds
            
        Returns:
            Tuple of (raw_response, formatted_text, process_steps)
        """
        if not self.initialized or self.agent is None:
            return (
                {"error": "🚫 Agent has not been initialized."},
                "🚫 Agent has not been initialized.",
                []
            )
            
        try:
            # Convert relative years to absolute
            processed_query = self.convert_relative_years_to_absolute(query, current_year=2025)
            
            # Create empty accumulator if not provided
            if callback_handler is None:
                empty_list = []
                callback_handler = (lambda _: None, empty_list)
                
            # Find decomposition tool
            decomp_tool = None
            for tool_name in self.tool_names:
                if tool_name.lower() == "decompose_query":
                    # Get the raw tool from client
                    raw_tools = self.client.get_tools()
                    for t in raw_tools:
                        if t.name.lower() == "decompose_query":
                            decomp_tool = t
                            break
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
                                "description": processed_query,
                                "depends_on": [],
                                "execution_group": 1
                            }
                        ]
                    }
                }
            else:
                try:
                    decomp_result = await decomp_tool.ainvoke({"query": processed_query})
                    execution_plan = self.parse_decomposition_result(decomp_result)
                except Exception as e:
                    print(f"[ERROR] Decomposition failed: {e}")
                    execution_plan = {
                        "needs_decomposition": False,
                        "complexity_reason": f"Decomposition error: {e}",
                        "execution_plan": {
                            "type": "simple",
                            "tasks": [
                                {
                                    "id": "task_1",
                                    "description": processed_query,
                                    "depends_on": [],
                                    "execution_group": 1
                                }
                            ]
                        }
                    }

            print(f"\n{'='*60}")
            print(f"Question: {processed_query}")
            print(f"Decomposition needed: {execution_plan.get('needs_decomposition', 'unknown')}")
            print(f"Reason: {execution_plan.get('complexity_reason', 'N/A')}")
            print(f"{'='*60}")
            
            # Check if this is a simple case with one task
            exec_plan = execution_plan["execution_plan"]
            if (exec_plan.get("type") == "simple" and 
                len(exec_plan["tasks"]) == 1 and 
                not execution_plan.get("needs_decomposition", True)):
                
                print("Simple query detected - executing original question directly")
                
                # Execute original question directly without task decomposition
                try:
                    response, process_steps = await asyncio.wait_for(
                        self._execute_simple_query(processed_query, callback_handler),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                    return {"error": error_msg}, error_msg, []

                # Get final answer from accumulated_text
                callback_func, accumulated_text = callback_handler
                final_text = "".join(accumulated_text)
                
                if not final_text:
                    final_text = response.get("final_answer", "No response generated")
                
                # For simple cases, process steps will be basic
                if not process_steps:
                    process_steps = [
                        {"role": "human", "content": processed_query},
                        {"role": "assistant", "content": final_text}
                    ]
                
                return response, final_text, process_steps
                
            else:
                # Complex case - use task decomposition
                print("Complex query detected - using task decomposition")
                try:
                    all_task_results, all_trace_outputs, task_execution_info = await asyncio.wait_for(
                        self.execute_execution_plan(execution_plan["execution_plan"]),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                    return {"error": error_msg}, error_msg, []

                # Compose final answer
                composite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                trace_log = "\n\n--- Task Trace Separator ---\n\n".join(all_trace_outputs)
                final_answer = await self.compose_final_answer(
                    composite_llm, processed_query, all_task_results, trace_log, execution_plan
                )
                validation_result = await self.validate_final_answer(
                    composite_llm, processed_query, final_answer, trace_log, execution_plan
                )
                
                # Create process steps for task-based execution
                process_steps = []
                
                # Add initial question
                process_steps.append({
                    "role": "human", 
                    "content": processed_query
                })
                
                # Add task execution info
                if task_execution_info and "tasks" in task_execution_info:
                    for task_info in task_execution_info["tasks"]:
                        # Add task description
                        process_steps.append({
                            "role": "task_description",
                            "content": f"**Task {task_info['id']}**: {task_info['description']}"
                        })
                        
                        # Add task result
                        process_steps.append({
                            "role": "task_result", 
                            "content": task_info['result']
                        })
                
                # Add final composed answer
                process_steps.append({
                    "role": "assistant", 
                    "content": final_answer
                })
                
                # Update callback handler with final answer
                callback_func, accumulated_text = callback_handler
                accumulated_text.append(final_answer)
                
                response = {
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
                        "trace_log": trace_log,
                        "composite_llm_output": final_answer,
                        "validation_llm_output": validation_result
                    }
                }
                
                return response, final_answer, process_steps
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error occurred: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print the full error in console for debugging
            return {"error": error_msg}, error_msg, []
    

    
    def get_streaming_callback(self, text_placeholder):
        """Create a callback function for streaming responses."""
        accumulated_text = []

        def callback_func(message: dict):
            nonlocal accumulated_text
            message_content = message.get("content", None)

            if isinstance(message_content, AIMessageChunk):
                content = message_content.content
                
                # If content is a simple string
                if isinstance(content, str):
                    accumulated_text.append(content)
                    text_placeholder.markdown("".join(accumulated_text))
                # If content is in list form (mainly occurs in Claude models)
                elif isinstance(content, list) and len(content) > 0:
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            accumulated_text.append(item["text"])
                    text_placeholder.markdown("".join(accumulated_text))
            
            # ToolMessage is not included in the final result
            elif isinstance(message_content, ToolMessage):
                # We can optionally log tool responses, but they won't be shown to user
                pass
                
            return None

        return callback_func, accumulated_text 