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
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            pass

def print_message():
    """
    Displays chat history on the screen.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # Create assistant message container
            with st.chat_message("assistant", avatar="🤖"):
                # Display assistant message content
                st.markdown(message["content"])

                # Check if we have process steps for this message
                message_id = i
                process_key = f"process_steps_{message_id-1}"  # User message is right before assistant message
                if process_key in st.session_state and st.session_state[process_key]:
                    process_steps = st.session_state[process_key]
                    # Show process steps in expander
                    with st.expander("🔍 **View Step-by-Step Process**", expanded=False):
                        for idx, step in enumerate(process_steps):
                            if step["role"] == "human":
                                st.markdown(f"### 💬 User Question:")
                                st.info(step['content'])
                            elif step["role"] == "ai":
                                st.markdown(f"### ✅ Final Answer:")
                                st.success(step['content'])
                            elif step["role"] == "tool":
                                st.markdown(f"### 🔍 Raw Data Retrieved:")
                                if "," in step["content"] and any(x in step["content"].lower() for x in ['hqla', 'billions']):
                                    # Display raw CSV data in a more readable format
                                    csv_lines = step["content"].strip().split('\r\n')
                                    for line in csv_lines:
                                        st.code(line, language=None)
                                else:
                                    st.code(step['content'], language=None)
                            
                            # Add separator if not the last step
                            if idx < len(process_steps) - 1:
                                st.divider()

                # Check if the next message is tool call information
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # Display tool call information in the same container as an expander
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # Increment by 2 as we processed two messages together
                else:
                    i += 1  # Increment by 1 as we only processed a regular message
        elif message["role"] == "thought":
            # Display thought process in a unique container
            with st.chat_message("assistant", avatar="💭"):
                st.markdown(f"**Thought Process:**\n{message['content']}")
            i += 1
        else:
            # Skip assistant_tool messages as they are handled above
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder, thought_placeholder):
    """
    Creates a streaming callback function.
    """
    accumulated_text = []
    accumulated_tool = []
    accumulated_thought = []
    current_thought = []
    in_thought = False
    
    # Add storage for raw responses to show in process breakdown
    process_steps = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool, accumulated_thought, current_thought, in_thought, process_steps
        message_content = message.get("content", None)

        # Store step in process breakdown
        if message_content:
            if isinstance(message_content, ToolMessage):
                process_steps.append({
                    "role": "tool",
                    "content": str(message_content.content)
                })
            elif isinstance(message_content, AIMessageChunk):
                if hasattr(message_content, 'content') and message_content.content:
                    # Check if it's not a thought message
                    content = message_content.content
                    if isinstance(content, str) and not content.lstrip().lower().startswith("thought"):
                        process_steps.append({
                            "role": "ai",
                            "content": content
                        })

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content

            # If content is in list form (mainly occurs in Claude models)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # Process text type
                if message_chunk["type"] == "text":
                    text_chunk = message_chunk["text"]
                    
                    # Check if this is a thought process
                    if text_chunk.lstrip().lower().startswith("thought") or in_thought:
                        in_thought = True
                        current_thought.append(text_chunk)
                        
                        # Check if thought has ended
                        if "\n\nAction:" in text_chunk or "\n\nFinal Answer:" in text_chunk:
                            in_thought = False
                            accumulated_thought.append("".join(current_thought))
                            thought_placeholder.markdown("".join(accumulated_thought))
                            current_thought = []
                        else:
                            thought_placeholder.markdown("".join(current_thought))
                    else:
                        accumulated_text.append(text_chunk)
                        text_placeholder.markdown("".join(accumulated_text))
                
                # Process tool use type
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander(
                        "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
            
            # Process if content is a simple string
            elif isinstance(content, str):
                # Check if this is a thought process
                if content.lstrip().lower().startswith("thought") or in_thought:
                    in_thought = True
                    current_thought.append(content)
                    
                    # Check if thought has ended
                    if "\n\nAction:" in content or "\n\nFinal Answer:" in content:
                        in_thought = False
                        accumulated_thought.append("".join(current_thought))
                        thought_placeholder.markdown("".join(accumulated_thought))
                        current_thought = []
                    else:
                        thought_placeholder.markdown("".join(current_thought))
                else:
                    accumulated_text.append(content)
                    text_placeholder.markdown("".join(accumulated_text))
            
            # Process tool call related information
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            
            # Handle other tool-related data formats
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        
        # Process if it's a tool message (tool response)
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
                
        return None

    return callback_func, accumulated_text, accumulated_tool, accumulated_thought, process_steps

async def astream_graph(agent, input_dict, callback, config):
    """
    Streams the agent execution.
    """
    result = await agent.ainvoke(
        input_dict,
        config={**config, "return_messages": True}
    )
    
    # Process each message in the trace
    for message in result["messages"]:
        role = getattr(message, "role", getattr(message, "type", "assistant"))
        content = getattr(message, "content", str(message))
        
        # Skip empty messages
        if not content:
            continue
            
        # Create a dict structure that matches the expected callback format
        message_dict = {"content": message}
        callback(message_dict)
        
        # Add small delay to make streaming look natural
        await asyncio.sleep(0.05)
        
    return result

async def process_query(query, text_placeholder, tool_placeholder, thought_placeholder, process_placeholder, timeout_seconds=60):
    """
    Processes user questions and generates responses.
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text, accumulated_tool, accumulated_thought, process_steps = (
                get_streaming_callback(text_placeholder, tool_placeholder, thought_placeholder)
            )
            try:
                # Debug: Print the query and message format
                print(f"Processing query: {query}")
                print(f"Messages format: {{'messages': '{query}'}}")
                
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": query},  # Same format as mcp_client_wo_thought.py
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, "", [], []

            final_text = "".join(accumulated_text)
            final_tool = "".join(accumulated_tool)
            final_thought = "".join(accumulated_thought)
            
            # Look for raw table data in tool responses
            raw_table_data = None
            for step in process_steps:
                if step["role"] == "tool" and "," in step["content"] and any(x in step["content"].lower() for x in ['hqla', 'billions']):
                    raw_table_data = step["content"]
                    break
            
            # If we found HQLA data, use that as the final text instead of the generated response
            if raw_table_data and "hqla" in raw_table_data.lower():
                final_text = raw_table_data
                # Update text placeholder with raw table data
                text_placeholder.markdown(f"```\n{raw_table_data}\n```")
            
            # Make sure the final answer is in process steps
            # Sometimes the streaming callbacks miss the final answer
            if process_steps and process_steps[-1]["role"] != "ai" and final_text:
                process_steps.append({
                    "role": "ai", 
                    "content": final_text
                })
            
            # Add initial query to process steps
            process_steps_with_query = [{"role": "human", "content": query}]
            
            # Filter steps to focus on query-tool-answer flow
            cleaned_steps = []
            for step in process_steps:
                # Skip any incomplete/empty steps
                if not step.get("content"):
                    continue
                    
                # Special handling for tool responses - make them more readable
                if step["role"] == "tool":
                    if "," in step["content"] and any(x in step["content"].lower() for x in ['hqla', 'billions', 'in billions']):
                        # This is likely the raw table data
                        cleaned_steps.append(step)
                # Include AI responses (but not thoughts)
                elif step["role"] == "ai" and not step["content"].lstrip().lower().startswith("thought"):
                    # Final answer should be included
                    cleaned_steps.append(step)
            
            # Combine process steps
            process_steps_with_query.extend(cleaned_steps)
            
            # Display process steps (response trajectory) in a toggle/expander
            if process_steps_with_query:
                with process_placeholder.expander("🔍 **View Step-by-Step Process**", expanded=False):
                    for idx, step in enumerate(process_steps_with_query):
                        if step["role"] == "human":
                            st.markdown(f"### 💬 User Question:")
                            st.info(step['content'])
                        elif step["role"] == "ai":
                            st.markdown(f"### ✅ Final Answer:")
                            st.success(step['content'])
                        elif step["role"] == "tool":
                            st.markdown(f"### 🔍 Raw Data Retrieved:")
                            if "," in step["content"] and any(x in step["content"].lower() for x in ['hqla', 'billions']):
                                # Display raw CSV data in a more readable format
                                csv_lines = step["content"].strip().split('\r\n')
                                for line in csv_lines:
                                    st.code(line, language=None)
                            else:
                                st.code(step['content'], language=None)
                        
                        # Add separator if not the last step
                        if idx < len(process_steps_with_query) - 1:
                            st.divider()
            
            # Extract thought processes from the response
            thoughts = []
            for message in response.get("messages", []):
                content = getattr(message, "content", "")
                if isinstance(content, str) and content.lstrip().lower().startswith("thought"):
                    thoughts.append(content)
            
            return response, final_text, final_tool, final_thought, process_steps_with_query
        else:
            return (
                {"error": "🚫 Agent has not been initialized."},
                "🚫 Agent has not been initialized.",
                "",
                "",
                [],
            )
    except Exception as e:
        import traceback
        error_msg = f"❌ Error occurred during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, "", "", []

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
- "What is the aggregate rent expense of American Tower Corp in 2014?"
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
        
        # Create separate containers for different message types
        assistant_container = st.chat_message("assistant", avatar="🤖")
        with assistant_container:
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            process_placeholder = st.empty()  # New placeholder for process toggle
        
        # Create separate thought container (not nested) but hidden
        # We'll track thoughts but not display them to users
        thought_container = st.empty()  # Changed from st.chat_message to st.empty()
        thought_placeholder = st.empty()
        
        # Process the query
        resp, final_text, final_tool, final_thought, process_steps = (
            st.session_state.event_loop.run_until_complete(
                process_query(
                    user_query,
                    text_placeholder,
                    tool_placeholder,
                    thought_placeholder,
                    process_placeholder,
                    st.session_state.timeout_seconds,
                )
            )
        )
        
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            
            # Store process steps in session state for this message
            message_id = len(st.session_state.history)
            st.session_state[f"process_steps_{message_id}"] = process_steps
            
            # Add assistant response to history
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            
            # Add tool information to history if present
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            
            st.rerun()
    else:
        st.warning(
            "⚠️ Agent is not initialized. Please click the 'Initialize Agent' button in the left sidebar to initialize."
        ) 