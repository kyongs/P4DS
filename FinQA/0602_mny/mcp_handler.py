import asyncio
import os
import json
from typing import Dict, List, Any, Tuple, Callable, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig

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

        # Initialize MultiServerMCPClient
        self.client = MultiServerMCPClient(
            {
                "chroma": {"command": "python", "args": ["./servers/chroma_server.py"], "transport": "stdio"},
                "fin":    {"command": "python", "args": ["./servers/fin_server.py"],    "transport": "stdio"},
                "math":   {"command": "python", "args": ["./servers/math_server.py"],   "transport": "stdio"},
                "sqlite": {"command": "python", "args": ["./servers/sqlite_server.py"], "transport": "stdio"},
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
        
        # Create ReAct agent without any system prompt 
        self.agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
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
    
    async def process_query(self, 
                           query: str, 
                           callback_handler: Optional[Tuple[Callable, List[str]]] = None,
                           thread_id: str = None,
                           timeout_seconds: int = 60) -> Tuple[Dict[str, Any], str, List[Dict[str, str]]]:
        """
        Process a user query and generate response.
        
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
            # Format query correctly with HumanMessage
            query_msg = HumanMessage(content=query)
            
            # Create empty accumulator if not provided
            if callback_handler is None:
                empty_list = []
                callback_handler = (lambda _: None, empty_list)
                
            try:
                response, process_steps = await asyncio.wait_for(
                    self._astream_graph(
                        self.agent,
                        {"messages": [query_msg]},
                        callback_handler,
                        RunnableConfig(
                            recursion_limit=self.recursion_limit,
                            thread_id=thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, []

            # Get final answer from accumulated_text
            callback_func, accumulated_text = callback_handler
            final_text = "".join(accumulated_text)
            
            # If no text was accumulated but we have a response, try to extract it
            if not final_text and isinstance(response, dict):
                if "messages" in response:
                    # Get the last message from the response
                    for msg in reversed(response["messages"]):
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            final_text = msg.content
                            break
            
            # Still no text? Look directly in the response
            if not final_text and hasattr(response, "content"):
                final_text = response.content
                
            # Fallback if we still have no response
            if not final_text:
                final_text = "I processed your request but couldn't generate a proper response. Please try again."
                
            return response, final_text, process_steps
        except Exception as e:
            import traceback
            error_msg = f"❌ Error occurred: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print the full error in console for debugging
            return {"error": error_msg}, error_msg, []
    
    async def _astream_graph(self, agent, input_dict, callback_tuple, config):
        """Stream the agent execution."""
        # Unpack the callback tuple
        callback_func, accumulated_text = callback_tuple
        
        # Get the trace from the agent (to capture all intermediate steps)
        result = await agent.ainvoke(
            input_dict,
            config={
                **config, 
                "return_messages": True,
                "recursion_limit": self.recursion_limit,
                "configurable": {"thread_id": config.get("thread_id", "default_thread")},
            }
        )
        
        # Process each message in the trace
        final_answer = ""
        
        # Collect process steps - ensure this is completely reset for each query
        process_steps = []
        
        # Add the initial question
        if isinstance(input_dict, dict) and "messages" in input_dict and input_dict["messages"]:
            # Extract the query
            query_msg = input_dict["messages"][0]
            if hasattr(query_msg, "content"):
                process_steps.append({
                    "role": "human",
                    "content": query_msg.content
                })
        
        # Extract intermediate steps from the ReAct agent
        if isinstance(result, dict):
            # Check if we have a full trace available
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    # Extract action step (thought + action)
                    if hasattr(step, "action") and step.action:
                        action_str = f"Action: {step.action.tool}\nAction Input: {step.action.tool_input}"
                        process_steps.append({
                            "role": "action",
                            "content": action_str
                        })
                    
                    # Extract observation step (tool output)
                    if hasattr(step, "observation"):
                        process_steps.append({
                            "role": "tool",
                            "content": str(step.observation)
                        })
                        
            # Process final messages
            if "messages" in result:
                for message in result["messages"]:
                    role = getattr(message, "role", getattr(message, "type", "assistant"))
                    content = getattr(message, "content", str(message))
                    
                    # Skip empty messages
                    if not content:
                        continue
                    
                    # Try to extract thought and action from assistant messages
                    if role == "assistant" and isinstance(content, str):
                        # Check for ReAct format in assistant messages
                        if "Thought:" in content and "Action:" in content:
                            # Extract thought
                            thought_section = content.split("Action:")[0]
                            if "Thought:" in thought_section:
                                thought_content = thought_section.split("Thought:")[1].strip()
                                if thought_content:
                                    process_steps.append({
                                        "role": "thought",
                                        "content": thought_content
                                    })
                            
                            # Extract action
                            if "Action:" in content and "Action Input:" in content:
                                try:
                                    action_part = content.split("Action:")[1].split("Action Input:")[0].strip()
                                    action_input_parts = content.split("Action Input:")
                                    if len(action_input_parts) > 1:
                                        input_part = action_input_parts[1].strip()
                                        process_steps.append({
                                            "role": "action",
                                            "content": f"Action: {action_part}\nAction Input: {input_part}"
                                        })
                                except Exception as e:
                                    print(f"Error parsing action: {e}")
                        
                        # Store final answer (last assistant message)
                        final_answer = content
                        # Add to process steps if it's meaningful and doesn't look like a ReAct format message
                        if content.strip() and not ("Thought:" in content or "Action:" in content):
                            process_steps.append({
                                "role": "ai",
                                "content": content
                            })
                    
                    # Store tool responses that weren't already captured
                    if role == "tool" and isinstance(content, str):
                        # Check if this tool message is already in process_steps to avoid duplication
                        is_duplicate = False
                        for step in process_steps:
                            if step["role"] == "tool" and step["content"] == content:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            process_steps.append({
                                "role": "tool",
                                "content": content
                            })
                    
                    # Extract observations from ReAct format
                    if role == "assistant" and isinstance(content, str) and "Observation:" in content:
                        observation_parts = content.split("Observation:")
                        if len(observation_parts) > 1:
                            observation = observation_parts[1].strip()
                            # Stop at the next section if it exists
                            for marker in ["Thought:", "Action:"]:
                                if marker in observation:
                                    observation = observation.split(marker)[0].strip()
                            
                            if observation:
                                is_duplicate = False
                                for step in process_steps:
                                    if step["role"] == "tool" and step["content"] == observation:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    process_steps.append({
                                        "role": "tool",
                                        "content": observation
                                    })
                    
                    # Other message types (like "system" or custom types)
                    if role not in ["assistant", "tool", "human"] and isinstance(content, str):
                        process_steps.append({
                            "role": role,
                            "content": content
                        })
                        
                    # Create a dict structure that matches the expected callback format
                    message_dict = {"content": message}
                    callback_func(message_dict)
                    
                    # Add small delay to make streaming look natural
                    await asyncio.sleep(0.05)
        
        # If we have a final answer but the callback didn't process it correctly,
        # force update the text placeholder through the accumulated_text
        if final_answer and not "".join(accumulated_text):  # If accumulated_text is empty
            accumulated_text.append(final_answer)
        
        # Sort process steps into a logical order
        if process_steps:
            # Define role priority (lower value = appears earlier)
            role_priority = {
                "human": 0,     # User question appears first
                "thought": 1,   # Thinking process appears next 
                "action": 2,    # Tool selection comes after thought
                "tool": 3,      # Tool output comes after action
                "ai": 4,        # Final answer appears last
            }
            
            # Create a key for sorting based on role priority and order of appearance
            def get_step_sort_key(idx, step):
                role = step["role"]
                # Get priority from the map, default to 99 for unknown roles
                priority = role_priority.get(role, 99)
                # Return tuple of (priority, original_index) to maintain original order for same priority
                return (priority, idx)
            
            # Sort process steps while preserving their order within the same role
            process_steps_with_idx = [(idx, step) for idx, step in enumerate(process_steps)]
            process_steps_with_idx.sort(key=lambda x: get_step_sort_key(*x))
            process_steps = [step for _, step in process_steps_with_idx]
        else:
            # If no detailed steps were extracted, at least include the initial question and final answer
            if final_answer:
                # User question should already be in process_steps from earlier in the function
                process_steps.append({
                    "role": "ai",
                    "content": final_answer
                })
                print("Warning: No detailed intermediate steps were captured. Only showing question and answer.")
        
        return result, process_steps
    
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