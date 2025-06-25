# prompt-engineering-2025spring-FinQA

Skeleton code repository for the Topic: Financial QA - MCP with Multiple Servers

## 🎯 Quick Start - Automated Pipeline

### Run Complete Pipeline
```bash
# Basic execution (auto-generates file names)
./run_pipeline.sh

# Custom suffix
./run_pipeline.sh experiment1

# Custom CPU usage (50%)
./run_pipeline.sh v1 0.5

# Auto suffix with custom CPU usage
./run_pipeline.sh "" 0.8
```

### File Organization
The pipeline automatically organizes files by date:
- **Today's results**: `./results/v{MMDD}/results_thoughts_v{N}.json`
- **Today's scores**: `./scores/v{MMDD}/results_with_score_v{N}.json`

Example for January 3rd:
- `./results/v0103/results_thoughts_v1.json`
- `./scores/v0103/results_with_score_v1.json`

## 📊 Individual Script Usage

### Question Processing
```bash
# Basic usage
python mcp_client_with_thought.py

# Custom configuration
python mcp_client_with_thought.py \
  --input ./data/qa_dict.json \
  --output ./results/v0103/custom_results.json \
  --cpu-usage 0.8 \
  --model gpt-4

# Show all options
python mcp_client_with_thought.py --help
```

### Scoring
```bash
# Basic usage
python score.py

# Custom configuration  
python score.py \
  --input-qa ./data/qa_dict_levels.json \
  --input-results ./results/v0103/results_thoughts_v1.json \
  --output ./scores/v0103/custom_scores.json \
  --cpu-usage 0.8

# Show all options
python score.py --help
```

## File Structure

**Data:**
- `data/test_db`: Pre-built Chroma DB (Vector DB) for the test set of FinQA
- `data/companies.csv`: Company information data which includes stock market status
- `data/companies.db`: companies.csv stored in SQLite DB
- `data/qa_dict.json`: QA set for the accuracy test, total 50 question and answer set 

**Servers:**
- `servers/chroma_server.py`: MCP server for the Chroma DB
- `servers/fin_server.py`: MCP server for financial calculations
- `servers/math_server.py`: MCP server for arithmetic calculations
- `servers/sqlite_server.py`: MCP server for the SQLite DB

**Main Scripts:**
- `mcp_client_with_thought.py`: Enhanced MCP client with multiprocessing support
- `score.py`: Enhanced scoring script with parallel evaluation
- `run_pipeline.sh`: Automated pipeline runner
- `mcp_client.py`: Legacy MCP client (deprecated)

## Requirements

```
uv >= 0.6.14, python >= 3.13
```

## Installation

```bash
$ uv venv
$ source .venv/bin/activate
$ uv pip install -r requirements.txt
```

## Set Environment

You should create a `.env` file in the root directory of the project. This file will contain your OpenAI API key.

```
OPENAI_API_KEY="[your_openai_api_key]"
```

## Project Structure

The code has been organized to separate the MCP server logic from the Streamlit UI:

- `mcp_handler.py`: Contains the core logic for MCP client initialization and query processing
- `streamlit_app.py`: Contains the Streamlit UI code that utilizes the MCP handler

This separation allows for easier maintenance and future enhancements.

## Running the Streamlit App

To run the Streamlit web interface:

```bash
$ streamlit run app/streamlit_app.py
```

This will launch a web server. Open your browser and navigate to `http://localhost:8501` to access the interface.

## Pre-defined Tool Examples

- `calculate_eps(net_income: float, outstanding_shares: int)`: Calculate the EPS of the company using net income and outstanding share
   - **Arguments**:
      - net_income: Net income value of the company
      - outstanding_shares: Total stock held by the company's shareholders
   - **Returns**:
      - EPS value or None if there is no value for arguments

- `calculate_cashflowfromoperations(net_income: float, non_cash_items: float, changes_in_working_capital: float)`: Calculate the cash flow from operations of the company using net income, non cash items and change in working capital
   - **Arguments**:
      - net_income: Net income value of the company
      - non_cash_items: Financial transactions or events that are recorded in a company's financial statements but do not involve the exchange of cash
      - changes_in_working_capital: Difference in a company's working capital between two reporting periods
   - **Returns**:
      - Value of cash flow from operations

- `retrieve_factual_data(question:str, ticker: str, fy: int) -> str`: Search vector DB for the financial reports with the question and ticker and fiscal year
   - **Arguments**:
      - question: Question need to be answered
      - ticker: Ticker of the company for filtering the documents
      - fy: Fiscal year for filtering the documents
   - **Returns**:
      - A related document for the question.

## References
- https://modelcontextprotocol.io/tutorials/building-mcp-with-llms
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/hannesrudolph/sqlite-explorer-fastmcp-mcp-server/tree/main
