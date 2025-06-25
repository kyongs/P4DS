# P4DS - Explainable Financial QA Framework

This project is an explainable AI framework for Financial Question Answering (QA) that leverages Model Context Protocol (MCP) to integrate with various servers for accurate and interpretable financial analysis.

## 📁 Project Structure

```
P4DS/
├── Explainable_Financial_QA/          # Main project (MCP-based Financial QA)
│   ├── app/                          # Streamlit web interface
│   ├── servers/                      # MCP servers
│   ├── data/                         # Database and data files
│   ├── results/                      # Experiment results
│   ├── scores/                       # Evaluation scores
│   └── ...
├── question_level_classification/     # Question difficulty classification
├── simple_finqa_eval/                # Simple FinQA evaluation
└── ...
```

## 🚀 Key Features

### 1. Explainable_Financial_QA (Main Project)
- **MCP-based Multi-Server Architecture**: ChromaDB, SQLite, Math calculation, Financial calculation servers
- **Interactive Streamlit Web Interface**: User-friendly QA interface with real-time processing
- **Automated Pipeline**: Complete process automation with `run_pipeline.sh`
- **Evaluation System**: Accuracy and explainability assessment

### 2. Question Level Classification
- **10-Level Difficulty Classification**: Automatic complexity classification of financial questions
- **GPT-4-based Classification**: Sophisticated prompt engineering

### 3. Simple FinQA Evaluation
- **Basic FinQA Evaluation**: Simple evaluation using GPT-4o-mini
- **DSL Parsing**: Domain Specific Language token parsing

## 🛠️ Installation and Setup

### 1. Basic Environment Setup
```bash
# Create and activate virtual environment
conda create -n finqa python=3.12
conda activate finqa

# Install basic dependencies
pip install -r requirements.txt
```

### 2. Explainable_Financial_QA Setup
```bash
cd Explainable_Financial_QA

# Using uv for dependency management (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 3. Environment Variables Setup
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 🎯 Usage

### 1. Running Explainable_Financial_QA

#### Interactive Streamlit Web Interface
```bash
cd Explainable_Financial_QA

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

**Streamlit App Features:**
- **Real-time Chat Interface**: Interactive conversation with the financial QA agent
- **Step-by-Step Process Visualization**: View detailed reasoning process with explanations
- **MCP Server Management**: Monitor and control connected MCP servers
- **System Settings**: Adjust timeout limits and recursion settings
- **Conversation History**: Maintain chat history across sessions
- **Explainability Features**: Detailed explanations with citations and source documents

**Example Questions:**
- "What was the hqla in the q4 of Citigroup in 2015?"
- "Which one's Operating Profit Margin is bigger between Air Products and Chemicals in 2014 and printing papers business of International Paper Company in 2006?"
- "What is the AON's decrease observed in the additions for tax positions of prior years as of 2018, in millions?"

#### Automated Pipeline Execution
```bash
cd Explainable_Financial_QA

# Basic execution (auto-generates file names)
./run_pipeline.sh

# Custom suffix
./run_pipeline.sh experiment1

# Custom CPU usage (50%)
./run_pipeline.sh v1 0.5
```

#### Individual Script Execution
```bash
# Question processing
python mcp_client_with_thought.py

# Score calculation
python score.py
```

### 2. Question Level Classification
```bash
cd question_level_classification

# Set INPUT_FILE and OUTPUT_FILE, then run
python question_level_classification.py
```

### 3. Simple FinQA Evaluation
```bash
cd simple_finqa_eval

# Run inference
python finqa_inference.py

# Run evaluation
cd evaluate
python evalute.py predict.json test.json
```

## 📊 MCP Server Configuration

### Available Servers
- **ChromaDB Server**: Vector database search for financial documents
- **SQLite Server**: Structured data querying for company information
- **Math Server**: Arithmetic calculations and mathematical operations
- **Financial Server**: Financial formulas and calculations (EPS, cash flow, etc.)
- **Google Search Server**: External information search capabilities
- **Decomposition Server**: Complex question decomposition into subtasks

### Pre-defined Tools
- `calculate_eps()`: EPS calculation
- `calculate_cashflowfromoperations()`: Cash flow from operations calculation
- `retrieve_factual_data()`: Related document search from vector DB
- `list_tables()`: List available database tables
- `read_query()`: Execute SQL queries
- `add()`, `subtract()`, `multiply()`, `divide()`: Mathematical operations

## 🌐 Streamlit Web Interface Details

### Main Features
1. **Interactive Chat Interface**
   - Real-time conversation with the financial QA agent
   - Support for complex financial questions
   - Automatic tool selection and execution

2. **Process Visualization**
   - Step-by-step reasoning process display
   - Tool call tracking and results
   - Task decomposition for complex queries
   - Detailed explanations with citations

3. **System Configuration**
   - Adjustable timeout settings (60-300 seconds)
   - Recursion limit controls (10-100 calls)
   - Real-time server status monitoring
   - Tool availability display

4. **Explainability Features**
   - Detailed step-by-step explanations
   - Source document citations
   - Reasoning process transparency
   - Evidence-based answers

### Interface Components
- **Main Chat Area**: Primary conversation interface
- **Sidebar**: System settings and server information
- **Process Viewer**: Expandable detailed process display
- **Server Status**: Real-time MCP server monitoring
- **Tool Documentation**: Available tools and their descriptions

### Usage Workflow
1. **Initialization**: Click "Initialize Agent" to start MCP servers
2. **Question Input**: Type financial questions in the chat input
3. **Processing**: Watch real-time processing with streaming responses
4. **Results**: View final answers with optional detailed process
5. **Explanation**: Access step-by-step reasoning and citations

## 📈 Evaluation and Analysis

### Automated Evaluation System
- **Accuracy Evaluation**: Comparison of predictions with ground truth
- **Explainability Evaluation**: Analysis of AI reasoning process
- **Difficulty-based Performance Analysis**: Performance differences by question complexity

### Result File Structure
```
results/v{MMDD}/results_thoughts_v{N}.json    # Inference results
scores/v{MMDD}/results_with_score_v{N}.json   # Evaluation scores
```

## 🔧 Technology Stack

### Core Technologies
- **Python 3.12+**: Main programming language
- **Model Context Protocol (MCP)**: Inter-server communication
- **OpenAI GPT-4**: Conversational AI model
- **Streamlit**: Web interface framework
- **ChromaDB**: Vector database
- **SQLite**: Relational database

### Key Libraries
- `mcp`: Model Context Protocol SDK
- `streamlit`: Web application framework
- `chromadb`: Vector database
- `langchain`: LLM integration framework
- `pandas`: Data processing
- `openai`: OpenAI API client
- `langgraph`: Workflow orchestration
- `asyncio`: Asynchronous programming

## 📚 References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FinQA GitHub](https://github.com/czyssrs/FinQA)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Report bugs or request features through issues
2. Contribute code through Pull Requests
3. Improve documentation and translations

## 📄 License

This project is developed for educational and research purposes.

---

**Development Team**: P4DS Team  
**Version**: 1.0.0  
**Last Updated**: June 25 2025