import streamlit as st
import asyncio
import nest_asyncio
import json
import os
from dotenv import load_dotenv, find_dotenv

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Load environment variables
_ = load_dotenv(find_dotenv())

# Helper function to display process steps (to avoid code duplication)
def display_process_steps(process_steps):
    for idx, step in enumerate(process_steps):
        if step["role"] == "human":
            st.markdown(f"### 👤 User:")
            st.info(step['content'])
        elif step["role"] == "ai":
            st.markdown(f"### 🤖 Assistant:")
            st.success(step['content'])
        elif step["role"] == "tool":
            st.markdown(f"### 🔧 Tool Result:")
            # Try to format tool output in a more readable way
            content = step['content'].strip()
            try:
                # First, try explicit JSON parsing
                if (content.startswith("{") and content.endswith("}")) or (content.startswith("[") and content.endswith("]")):
                    try:
                        json_content = json.loads(content)
                        st.json(json_content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, display as code
                        st.code(content, language=None)
                else:
                    # Check if it looks like an error message
                    if "ERROR" in content and "{" in content and "}" in content:
                        # Try to extract just the JSON part
                        try:
                            # Find JSON-like parts and try to parse them
                            start_idx = content.find("{")
                            end_idx = content.rfind("}") + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = content[start_idx:end_idx]
                                # Try to parse and display as JSON
                                json_content = json.loads(json_str)
                                st.error(f"Error: {json_content.get('message', 'Unknown error')}")
                                # Still show the full content as code
                                st.code(content, language=None)
                            else:
                                st.code(content, language=None)
                        except:
                            # If extraction fails, just show as code
                            st.code(content, language=None)
                    else:
                        # Regular output as code
                        st.code(content, language=None)
            except Exception as e:
                # Fallback for any unexpected errors
                st.warning(f"Error displaying result: {str(e)}")
                st.code(step['content'], language=None)
        elif step["role"] == "action":
            st.markdown(f"### 🛠️ Tool Call:")
            st.warning(step['content'])
        elif step["role"] == "thought":
            st.markdown(f"### 💭 Thinking Process:")
            st.info(step['content'])
        else:
            # Handle any other step types
            st.markdown(f"### ℹ️ {step['role'].capitalize()}:")
            st.text(step['content'])
        
        # Add separator if not the last step
        if idx < len(process_steps) - 1:
            st.divider()

# Initialize session state
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # Session initialization flag
    st.session_state.agent = None  # Storage for ReAct agent object
    st.session_state.history = []  # List for storing conversation history
    st.session_state.mcp_client = None  # Storage for MCP client object
    st.session_state.timeout_seconds = 120  # Response generation time limit (seconds)
    st.session_state.recursion_limit = 50  # Recursion call limit

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_" + str(hash(os.urandom(16)))

# --- Function Definitions ---

async def cleanup_mcp_client():
    """
    Safely terminates the existing MCP client.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            # Properly close any pending generators
            client = st.session_state.mcp_client
            
            # Get all generators and close them explicitly
            if hasattr(client, "_clients"):
                for server_name, server_client in client._clients.items():
                    if hasattr(server_client, "_generators"):
                        for gen in server_client._generators:
                            try:
                                if hasattr(gen, "aclose"):
                                    await gen.aclose()
                            except Exception as e:
                                print(f"Error closing generator in {server_name}: {e}")
            
            # Now try to exit the client properly
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
            
            # Force garbage collection to clean up any remaining references
            import gc
            gc.collect()
            
            print("MCP client successfully cleaned up")
        except Exception as e:
            import traceback
            print(f"Error during MCP client cleanup: {e}")
            print(traceback.format_exc())

def print_message():
    """
    Displays chat history on the screen (simplified version).
    """
    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
        elif message["role"] == "assistant":
            # Only display assistant messages (no tool call info or processes)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                
                # Check if we have process steps for this message
                if "process_key" in message and message["process_key"] in st.session_state:
                    process_steps = st.session_state[message["process_key"]]
                    # Show process steps in expander
                    with st.expander("🔍 **View Step-by-Step Process**", expanded=False):
                        # Display process steps
                        display_process_steps(process_steps)
                # For backward compatibility with old messages
                elif f"process_steps_{i}" in st.session_state:  
                    process_steps = st.session_state[f"process_steps_{i}"]
                    # Show process steps in expander
                    with st.expander("🔍 **View Step-by-Step Process**", expanded=False):
                        # Display process steps
                        display_process_steps(process_steps)

def get_streaming_callback(text_placeholder):
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

async def astream_graph(agent, input_dict, callback_tuple, config):
    """
    Streams the agent execution.
    """
    # Unpack the callback tuple
    callback_func, accumulated_text = callback_tuple
    
    # Get the trace from the agent (to capture all intermediate steps)
    result = await agent.ainvoke(
        input_dict,
        config={
            **config, 
            "return_messages": True,
            "recursion_limit": st.session_state.recursion_limit,
            "configurable": {"thread_id": st.session_state.thread_id},
        }
    )
    
    # Debug: Print result structure
    print(f"Agent result structure: {type(result)}")
    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")
        for key in result.keys():
            print(f"Key: {key}, Type: {type(result[key])}")
            
        if "messages" in result:
            print(f"Messages type: {type(result['messages'])}")
            print(f"Messages count: {len(result['messages'])}")
            for i, msg in enumerate(result['messages']):
                print(f"Message {i}: Type={type(msg)}, Role={getattr(msg, 'role', 'unknown')}")
                
        if "intermediate_steps" in result:
            print(f"Intermediate steps count: {len(result['intermediate_steps'])}")
            for i, step in enumerate(result['intermediate_steps']):
                print(f"Step {i}: Type={type(step)}, Has action={hasattr(step, 'action')}, Has observation={hasattr(step, 'observation')}")
    
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
                callback_func(message_dict)  # Use callback_func here, not the tuple
                
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

async def process_query(query, text_placeholder, timeout_seconds=60):
    """
    Processes user questions and generates responses (simplified version).
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text = get_streaming_callback(text_placeholder)
            
            try:
                # Format query correctly with HumanMessage
                query_msg = HumanMessage(content=query)
                
                # Debug: Print input structure
                print(f"Query input: {query_msg}")
                
                # Reset thread_id for a completely new conversation context
                # This ensures process steps don't accumulate between queries
                st.session_state.thread_id = "thread_" + str(hash(os.urandom(16)))
                
                response, process_steps = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [query_msg]},  # Use list with message object
                        callback_tuple=(streaming_callback, accumulated_text),  # Match parameter name in astream_graph
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, []

            # Check if we got a response in accumulated_text
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
        else:
            return (
                {"error": "🚫 Agent has not been initialized."},
                "🚫 Agent has not been initialized.",
                []
            )
    except Exception as e:
        import traceback
        error_msg = f"❌ Error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Print the full error in console for debugging
        return {"error": error_msg}, error_msg, []

async def initialize_session():
    """
    Initializes MCP session and agent.
    """
    with st.spinner("🔄 Connecting to MCP server..."):
        # First safely clean up existing client
        await cleanup_mcp_client()

        # Initialize MultiServerMCPClient with the same configuration as in mcp_client_wo_thought.py
        client = MultiServerMCPClient(
            {
                "chroma": {"command": "python", "args": ["./servers/chroma_server.py"], "transport": "stdio"},
                "fin":    {"command": "python", "args": ["./servers/fin_server.py"],    "transport": "stdio"},
                "math":   {"command": "python", "args": ["./servers/math_server.py"],   "transport": "stdio"},
                "sqlite": {"command": "python", "args": ["./servers/sqlite_server.py"], "transport": "stdio"},
            }
        )
        
        await client.__aenter__()
        tools = client.get_tools()
        
        # Print detailed information about the tools
        st.session_state.tool_count = len(tools)
        st.session_state.tool_names = []
        st.session_state.tools_info = []
        
        # Group tools by server
        tool_servers = {}
        for tool in tools:
            # Extract server name from tool name (before the dot)
            if "." in tool.name:
                server_name = tool.name.split(".")[0]
                if server_name not in tool_servers:
                    tool_servers[server_name] = []
                tool_servers[server_name].append(tool.name)
            else:
                if "unknown" not in tool_servers:
                    tool_servers["unknown"] = []
                tool_servers["unknown"].append(tool.name)
            
            # Store tool name
            st.session_state.tool_names.append(tool.name)
            # Store detailed tool info
            st.session_state.tools_info.append(f"Tool: {tool.name} - {tool.description}")
        
        # Store server summary
        st.session_state.server_summary = []
        for server, tool_list in tool_servers.items():
            st.session_state.server_summary.append(f"Server: {server} - {len(tool_list)} tools")
            for tool_name in tool_list:
                st.session_state.server_summary.append(f"  - {tool_name}")
        
        st.session_state.mcp_client = client

        # Initialize OpenAI model
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )
        
        # Create ReAct agent without any system prompt 
        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
        )
        
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True

# --- Setup Streamlit UI ---

st.set_page_config(page_title="FinQA Agent", page_icon="💰", layout="wide")

# Page title and description
st.title("💬 Financial Question Answering Agent")
st.markdown("""
✨ Ask financial questions to this ReAct agent that utilizes MCP tools.

**Example questions you can ask:**
- "What was the hqla in the q4 of Citigroup in 2015?"
- "in 2011 what was the SL Green Realty Corp's percent of the change in the account balance at end of year"
- "What is the long-term component of BlackRock at 12/31/2011?"
""")

# Sidebar configuration
with st.sidebar:
    st.subheader("⚙️ System Settings")
    
    # Timeout setting slider
    st.session_state.timeout_seconds = st.slider(
        "⏱️ Response generation time limit (seconds)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="Set the maximum time for the agent to generate a response. Complex tasks may require more time.",
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ Recursion call limit (count)",
        min_value=10,
        max_value=100,
        value=st.session_state.recursion_limit,
        step=10,
        help="Set the recursion call limit. Setting too high a value may cause memory issues.",
    )
    
    st.divider()  # Add divider
    
    # System information
    st.subheader("📊 System Information")
    st.write(
        f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}"
    )
    st.write("🧠 Current Model: gpt-4o-mini")
    
    # Add server summary
    if st.session_state.session_initialized and "server_summary" in st.session_state:
        with st.expander("MCP Servers & Tools", expanded=True):
            for line in st.session_state.server_summary:
                st.write(line)
    
    st.divider()  # Add divider
    
    # Initialize button
    if st.button(
        "Initialize Agent",
        key="init_button",
        type="primary",
        use_container_width=True,
    ):
        # Initialize the agent
        success = st.session_state.event_loop.run_until_complete(initialize_session())
        
        if success:
            st.success("✅ Agent has been initialized.")
        else:
            st.error("❌ Failed to initialize agent.")
        
        # Refresh page
        st.rerun()
    
    # Reset conversation button
    if st.button("Reset Conversation", use_container_width=True):
        # Reset thread_id
        st.session_state.thread_id = "thread_" + str(hash(os.urandom(16)))
        
        # Reset conversation history
        st.session_state.history = []
        
        # Cleanup MCP client properly
        st.session_state.event_loop.run_until_complete(cleanup_mcp_client())
        
        # Notification message
        st.success("✅ Conversation has been reset.")
        
        # Refresh page
        st.rerun()

# --- Initialize default session (if not initialized) ---
if not st.session_state.session_initialized:
    st.info(
        "Agent is not initialized. Please click the 'Initialize Agent' button in the left sidebar to initialize."
    )

# --- Print conversation history ---
print_message()

# --- User input and processing ---
user_query = st.chat_input("💬 Enter your financial question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        
        # Use only a single container
        assistant_container = st.chat_message("assistant", avatar="🤖")
        with assistant_container:
            text_placeholder = st.empty()
            process_placeholder = st.empty()  # New placeholder for process display
        
        # Simplified process_query call
        resp, final_text, process_steps = st.session_state.event_loop.run_until_complete(
            process_query(
                user_query,
                text_placeholder,
                st.session_state.timeout_seconds,
            )
        )
        
        if "error" in resp:
            st.error(resp["error"])
        else:
            # Store in history
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            
            # Store process steps in session state for this message
            message_id = len(st.session_state.history) - 1  # Index of the assistant message
            process_key = f"process_steps_{st.session_state.thread_id}_{message_id}"
            st.session_state[process_key] = process_steps
            
            # Update the process_key in the history for reference
            st.session_state.history[-1]["process_key"] = process_key
            
            # Display process steps (optional, can remove if you want to rely only on print_message)
            with process_placeholder.expander("🔍 **View Step-by-Step Process**", expanded=False):
                display_process_steps(process_steps)
            
            st.rerun()
    else:
        st.warning("⚠️ Agent is not initialized. Please click the 'Initialize Agent' button in the left sidebar first.") 