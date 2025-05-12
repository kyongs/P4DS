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
                process_key = f"process_steps_{i}"
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
                                st.code(step['content'], language=None)
                            
                            # Add separator if not the last step
                            if idx < len(process_steps) - 1:
                                st.divider()

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
    
    result = await agent.ainvoke(
        input_dict,
        config={**config, "return_messages": True}
    )
    
    # Debug: Print result structure
    print(f"Agent result structure: {type(result)}")
    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")
        if "messages" in result:
            print(f"Messages type: {type(result['messages'])}")
            print(f"Messages count: {len(result['messages'])}")
    
    # Process each message in the trace
    final_answer = ""
    
    # Collect process steps
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
    
    if isinstance(result, dict) and "messages" in result:
        for message in result["messages"]:
            role = getattr(message, "role", getattr(message, "type", "assistant"))
            content = getattr(message, "content", str(message))
            
            # Skip empty messages
            if not content:
                continue
                
            # Store final answer (last assistant message)
            if role == "assistant" and isinstance(content, str):
                final_answer = content
                # Add to process steps if it's meaningful
                if content.strip():
                    process_steps.append({
                        "role": "ai",
                        "content": content
                    })
            
            # Store tool responses in process steps
            if role == "tool" and isinstance(content, str):
                process_steps.append({
                    "role": "tool",
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
    
    # Return result and process steps
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
            st.session_state[f"process_steps_{message_id}"] = process_steps
            
            # Display process steps (optional, can remove if you want to rely only on print_message)
            with process_placeholder.expander("🔍 **View Step-by-Step Process**", expanded=False):
                for idx, step in enumerate(process_steps):
                    if step["role"] == "human":
                        st.markdown(f"### 💬 User Question:")
                        st.info(step['content'])
                    elif step["role"] == "ai":
                        st.markdown(f"### ✅ Final Answer:")
                        st.success(step['content'])
                    elif step["role"] == "tool":
                        st.markdown(f"### 🔍 Raw Data Retrieved:")
                        st.code(step['content'], language=None)
                    
                    # Add separator if not the last step
                    if idx < len(process_steps) - 1:
                        st.divider()
            
            st.rerun()
    else:
        st.warning("⚠️ Agent is not initialized. Please click the 'Initialize Agent' button in the left sidebar first.") 