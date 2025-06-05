# servers/decomposition_server.py

from typing import Annotated
from mcp.server.fastmcp import FastMCP
import openai

mcp = FastMCP("Decompose")

@mcp.tool(description="Decompose a financial reasoning query into 3 key subtasks.")
def decompose_query(
    query: Annotated[str, "The financial question to break into subtasks."]
) -> list:
    """
    Uses a few-shot prompt to show examples of how to decompose various
    levels of question complexity into three simple subtasks.
    """

    few_shot_prompt = r"""
You are an expert at breaking down financial queries into 3 key sub‐tasks. 
For each example below, list exactly three lines (one subtask per line) with no extra commentary.

Example 1 (Simple “factual lookup”):
Question: What was the HQLA in Q4 of Citigroup in 2015?
Subtasks:
1. Retrieve Citigroup’s HQLA value as of December 31, 2015 from the database.
2. (No calculation needed—this is a direct lookup.)
3. Return that HQLA value (e.g., “378.5 billion dollars”).

Example 2 (Single-step “factual lookup” + small transformation):
Question: What was the port call costs of Royal Caribbean Cruises in 2012?
Subtasks:
1. Retrieve Royal Caribbean Cruises’ “port costs included in passenger ticket revenues” for the full year 2012.
2. (No arithmetic—this is a direct field in the 10-K.)
3. Return that value (e.g., “459.8 million dollars”).

Example 3 (“numerical calculation”):
Question: What is the aggregate rent expense of American Tower Corp in 2014?
Subtasks:
1. Retrieve “aggregate rent expense (including straight-line rent)” for American Tower Corp for the year ended December 31, 2014.
2. (No additional arithmetic—this field is already provided as a single line in the 10-K.)
3. Return that rent expense (e.g., “655.0 million dollars”).

Example 4 (“year-relative lookup” + direct lookup):
Question: From the perspective of 6 years ago, what percentage of total minimum lease payments of Dish Network are due in 2015?
Subtasks:
1. Convert “6 years ago” to an absolute year (if today is 2021, that becomes 2015).
2. Retrieve Dish Network’s “percentage of total minimum lease payments due in 2015” from the 10-K as of 2015.
3. Return that percentage (e.g., “9.772%”).

Example 5 (“multi-step reasoning” requiring two retrievals + a subtraction):
Question: What is the difference of Operating Profit Margin between Air Products and Chemicals in 2016 and printing papers business of International Paper Company in 2006?
Subtasks:
1. Retrieve Air Products and Chemicals’ “Operating Profit Margin” for the year 2016.
2. Retrieve International Paper Company’s “Operating Profit Margin of the printing papers business” for the year 2006.
3. Subtract the 2006 IP printing-papers margin from the 2016 Air Products margin and return the difference (e.g., “17.00441774”).

Example 6 (“multi-step reasoning” requiring two lookup+calculations and an average):
Question: What is the average of current ratio between American Tower Corp in 2012 and DISH Network Corporation in 2011?
Subtasks:
1. Retrieve American Tower Corp’s “current assets” and “current liabilities” for 2012; compute its current ratio (current assets ÷ current liabilities).
2. Retrieve DISH Network Corporation’s “current assets” and “current liabilities” for 2011; compute its current ratio.
3. Compute the average of these two ratios and return that number (e.g., “1.786341177”).

Example 7 (“multi-tiered multi-tool reasoning”):
Question: What is Goldman Sachs Group’s total Assets change from 2014 to 2016 multiplied by the average of current ratio between American Tower Corp in 2012 and DISH Network Corporation in 2011?
Subtasks:
1. Retrieve Goldman Sachs Group’s “Total Assets” as of December 31, 2014 and as of December 31, 2016; compute ΔAssets = Assets(2016) − Assets(2014).
2. Retrieve American Tower Corp’s “current assets” & “current liabilities” for 2012, compute that current ratio; retrieve DISH Network’s “current assets” & “current liabilities” for 2011, compute that ratio; then compute the average of those two ratios.
3. Multiply ΔAssets (from step 1) by the average current ratio (from step 2) and return the product (e.g., “48,722.45 million”).

Now decompose this new question into exactly three steps (one subtask per line):

Question: {query}
Subtasks:
"""

    # Call the OpenAI ChatCompletion endpoint (or whichever LLM you use)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a query decomposition assistant specialized in financial analysis."},
            {"role": "user",   "content": few_shot_prompt.format(query=query)}
        ],
        temperature=0.0,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    content = response["choices"][0]["message"]["content"].strip()
    # Split into lines; return a Python list of the three subtasks
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return lines

if __name__ == "__main__":
    mcp.run(transport="stdio")
