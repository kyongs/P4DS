import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import atexit
from dotenv import load_dotenv, find_dotenv
import pandas as pd

# Import the MCPHandler
from mcp_handler import MCPHandler

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

# Load environment variables
_ = load_dotenv(find_dotenv())

# Helper function to display process steps (to avoid code duplication)
def display_process_steps(process_steps):
    # Use columns to organize content types
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
            try:
                # Check if it's valid JSON
                content = step['content'].strip()
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
    st.session_state.history = []  # List for storing conversation history
    st.session_state.mcp_handler = MCPHandler()  # Initialize the MCP handler
    st.session_state.timeout_seconds = 120  # Response generation time limit (seconds)
    st.session_state.recursion_limit = 50  # Recursion call limit

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_" + str(hash(os.urandom(16)))

# Register cleanup handler for Streamlit session end
def cleanup_on_app_close():
    """Cleans up resources when the Streamlit app is closing."""
    if "event_loop" in st.session_state and "mcp_handler" in st.session_state:
        try:
            st.session_state.event_loop.run_until_complete(
                st.session_state.mcp_handler.cleanup_client()
            )
        except Exception as e:
            print(f"Error during shutdown cleanup: {e}")
            
    print("Application cleanup completed")

# Register the cleanup handler
atexit.register(cleanup_on_app_close)

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
                    # Show process steps in expander - avoid nesting expanders when display_process_steps is called
                    if st.button("🔍 **View Step-by-Step Process**", key=f"process_btn_{i}"):
                        st.session_state[f"show_process_{i}"] = True
                    
                    # Display the process steps if button was clicked
                    if f"show_process_{i}" in st.session_state and st.session_state[f"show_process_{i}"]:
                        # Clear button
                        if st.button("Hide Details", key=f"hide_process_{i}"):
                            st.session_state[f"show_process_{i}"] = False
                            st.rerun()
                        
                        # Display process steps
                        display_process_steps(process_steps)
                # For backward compatibility with old messages
                elif f"process_steps_{i}" in st.session_state:  
                    process_steps = st.session_state[f"process_steps_{i}"]
                    # Show process steps with button instead of expander
                    if st.button("🔍 **View Step-by-Step Process**", key=f"old_process_btn_{i}"):
                        st.session_state[f"show_old_process_{i}"] = True
                    
                    # Display the process steps if button was clicked
                    if f"show_old_process_{i}" in st.session_state and st.session_state[f"show_old_process_{i}"]:
                        # Clear button
                        if st.button("Hide Details", key=f"hide_old_process_{i}"):
                            st.session_state[f"show_old_process_{i}"] = False
                            st.rerun()
                        
                        # Display process steps
                        display_process_steps(process_steps)

async def initialize_session():
    """
    Initializes MCP session and agent.
    """
    with st.spinner("🔄 Connecting to MCP server..."):
        # Initialize the MCP handler
        success = await st.session_state.mcp_handler.initialize()
        if success:
            # Copy some information to session state for UI display
            st.session_state.tool_count = st.session_state.mcp_handler.tool_count
            st.session_state.tool_names = st.session_state.mcp_handler.tool_names
            st.session_state.tools_info = st.session_state.mcp_handler.tools_info
            st.session_state.server_tools = st.session_state.mcp_handler.server_tools
            
            # Set session recursion limit
            st.session_state.mcp_handler.recursion_limit = st.session_state.recursion_limit
            
            # Mark session as initialized
            st.session_state.session_initialized = True
        return success

async def process_user_query(user_query, text_placeholder, timeout_seconds=60):
    """Process a user query and handle UI updates."""
    
    # Get streaming callback for real-time UI updates
    streaming_callback = st.session_state.mcp_handler.get_streaming_callback(text_placeholder)
    
    # Process the query through the MCP handler
    resp, final_text, process_steps = await st.session_state.mcp_handler.process_query(
        query=user_query,
        callback_handler=streaming_callback,
        thread_id=st.session_state.thread_id,
        timeout_seconds=timeout_seconds
    )
    
    # Handle errors
    if "error" in resp:
        return resp, final_text, process_steps
    
    # Store in history
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": final_text})
    
    # Store process steps in session state for this message
    message_id = len(st.session_state.history) - 1  # Index of the assistant message
    process_key = f"process_steps_{st.session_state.thread_id}_{message_id}"
    st.session_state[process_key] = process_steps
    
    # Update the process_key in the history for reference
    st.session_state.history[-1]["process_key"] = process_key
    
    return resp, final_text, process_steps

def main():
    """Main function to setup the Streamlit UI."""
    
    st.set_page_config(page_title="FinQA Agent", page_icon="💰", layout="wide")

    # Page title and description
    st.title("💬 Financial Question Answering Agent")
    st.markdown("""
    ✨ Ask financial questions to this ReAct agent that utilizes MCP tools.

    **Example questions you can ask:**
    - "What was the hqla in the q4 of Citigroup in 2015?"
    - "In 2011 what was the SL Green Realty Corp's percent of the change in the account balance at end of year"
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
        
        # Update recursion limit in handler if it's already initialized
        if st.session_state.session_initialized:
            st.session_state.mcp_handler.recursion_limit = st.session_state.recursion_limit
        
        st.divider()  # Add divider
        
        # System information
        st.subheader("📊 System Information")
        st.write(
            f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}"
        )
        st.write("🧠 Current Model: gpt-4o-mini")
        
        # Add server summary
        if st.session_state.session_initialized and "server_tools" in st.session_state:
            st.subheader("🔌 Available MCP Servers & Tools")
            
            # Server icon and description mapping for better UI
            server_icons = {
                "chroma": "🔍",   # Search/retrieval
                "fin": "💰",      # Finance
                "math": "🧮",     # Math
                "sqlite": "🗃️",   # Database
                "unknown": "🔧"   # Unknown tools
            }
            
            # Display summary as compact badges
            summary_cols = st.columns([1, 1])
            with summary_cols[0]:
                # Count total number of available tools
                total_tool_count = sum(len(tools) for tools in st.session_state.server_tools.values())
                st.markdown(f"**Tools**: {total_tool_count}")
                
            with summary_cols[1]:
                # Get count of active servers
                non_empty_servers = [s for s in st.session_state.server_tools.keys() 
                                  if s in st.session_state.server_tools and len(st.session_state.server_tools[s]) > 0]
                st.markdown(f"**Servers**: {len(non_empty_servers)}")
            
            # Create simpler UI with cleaner tool display
            for server_name in ["chroma", "fin", "math", "sqlite", "unknown"]:
                # Skip if server doesn't exist in our data
                if server_name not in st.session_state.server_tools:
                    continue
                    
                tools = st.session_state.server_tools[server_name]
                if len(tools) == 0:
                    continue  # Skip empty servers
                    
                # Get icon
                icon = server_icons.get(server_name, "🔧")
                
                # Create expander with server info
                with st.expander(f"{icon} {server_name.capitalize()} Server ({len(tools)} tools)", expanded=True):
                    # Sort tools by name
                    sorted_tools = sorted(tools, key=lambda x: x["name"])
                    
                    # Create a simple markdown list
                    tool_list = []
                    for tool in sorted_tools:
                        tool_list.append(f"• **{tool['name']}**")
                    
                    # Join with line breaks and display
                    st.markdown("  \n".join(tool_list))
                    
                    # Add button for viewing full details
                    if st.button(f"View Documentation", key=f"full_{server_name}"):
                        st.session_state[f"show_full_{server_name}"] = True
                    
                    # Show full details with examples and full descriptions
                    if f"show_full_{server_name}" in st.session_state and st.session_state[f"show_full_{server_name}"]:
                        st.markdown("### Tool Documentation")
                        for tool in sorted_tools:
                            st.markdown(f"#### {tool['name']}")
                            st.markdown(f"{tool['description']}")
                            st.divider()
                        
                        # Hide button
                        if st.button(f"Hide Documentation", key=f"hide_full_{server_name}"):
                            st.session_state[f"show_full_{server_name}"] = False
                            st.rerun()
        
        st.divider()  # Add divider
        
        # Initialize button
        if st.button(
            "Initialize Agent",
            key="init_button",
            type="primary",
            use_container_width=True,
        ):
            # Initialize the agent
            try:
                success = st.session_state.event_loop.run_until_complete(initialize_session())
                
                if success:
                    st.success("✅ Agent has been initialized.")
                else:
                    st.error("❌ Failed to initialize agent.")
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                st.error(f"❌ Error during initialization: {str(e)}")
                st.expander("Technical Details", expanded=False).code(error_traceback)
                print(f"Initialization error: {str(e)}\n{error_traceback}")
            
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
            
            # Process the query
            resp, final_text, process_steps = st.session_state.event_loop.run_until_complete(
                process_user_query(
                    user_query,
                    text_placeholder,
                    st.session_state.timeout_seconds,
                )
            )
            
            if "error" in resp:
                st.error(resp["error"])
            else:
                # Display process steps (optional)
                with process_placeholder:
                    if st.button("🔍 **View Step-by-Step Process**", key=f"inline_process"):
                        st.session_state["show_inline_process"] = True
                    
                    # Display the process steps if button was clicked
                    if "show_inline_process" in st.session_state and st.session_state["show_inline_process"]:
                        # Hide button
                        if st.button("Hide Details", key="hide_inline_process"):
                            st.session_state["show_inline_process"] = False
                            st.rerun()
                        
                        # Display process steps
                        display_process_steps(process_steps)
                
                st.rerun()
        else:
            st.warning("⚠️ Agent is not initialized. Please click the 'Initialize Agent' button in the left sidebar first.")

if __name__ == "__main__":
    main()