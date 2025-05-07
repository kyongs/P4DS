# prompt-engineering-2025spring-FinQA

Skeleton code repository for the Topic: Financial QA - MCP with Multiple Servers

**File list**
- data
  - test_db: Pre-built Chroma DB (Vector DB) for the test set of FinQA
  - companies.csv: Company information data which includes stock market status
  - companies.db: companies.csv stored in SQLite DB
  - qa_dict.json: QA set for the accuracy test, total 50 question and answer set 
- servers
  - chroma_server.py: MCP server for the Chroma DB
  - fin_server.py: MCP server for financial calculations
  - math_server.py: MCP server for arithmetic calculations
  - sqlite_server.py: MCP server for the SQLite DB
- mcp_client.py: MCP client, run this code to generate result for the questions
- score.py: Run this code for scoring the accuracy with your result 

## References
- https://modelcontextprotocol.io/tutorials/building-mcp-with-llms
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/hannesrudolph/sqlite-explorer-fastmcp-mcp-server/tree/main

## Requirements

```
uv >= 0.6.14, python >= 3.13
```

## Installation

```
$ uv venv
$ uv sync
```

## Set Environment

You should create a `.env` file in the root directory of the project. This file will contain your OpenAI API key.

```
OPENAI_API_KEY="[your_openai_api_key]"
```

## Run MCP Client and Get Accuracy

```
$ python mcp_client.py
$ python score.py
```

### Pre-defined Tool Examples

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
