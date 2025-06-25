# servers/decomposition_server.py

from typing import Annotated
from mcp.server.fastmcp import FastMCP
import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

mcp = FastMCP("Decompose")

@mcp.tool(description="Decompose a financial reasoning query into multiple subtasks.")
def decompose_query(
    query: Annotated[str, "The financial question to break into subtasks."]
) -> list:
    """
    Uses a few-shot prompt to show examples of how to decompose various
    levels of question complexity into three simple subtasks.
    """

    few_shot_prompt = f"""
You are a financial query decomposition assistant.
Your job is to classify a financial question into one of the four categories and break it down into step-by-step subtasks.

If the question asks for structured metadata (e.g., Symbol, Price, Market cap, Volume, P/E, Sector, Headquarters Location, Founded, etc.),
you can retrieve each value directly from the database using sqlite_server.

There are 4 categories:

1. Simple retrieval: No calculation, just retrieve the value directly (for one year, one company).
2. Simple calculation: Requires a financial ratio formula (for one year, one company).
3. Two-year calculation: One company, multiple years, using arithmetic or logical comparison (e.g., difference, average).
4. Two-company calculation: Comparing or combining values from two companies (may include arithmetic or logical comparison).

Each output must include:
- Step 1: ...
- Step 2: ...
- More steps (if needed)

---

Example 1: Simple retrieval
Question: What was the HQLA in the Q4 of Citigroup in 2015?

Subtasks:
Step 1: Retrieve the HQLA of Citigroup in Q4 2015

---

Example 2: Simple calculation
Question: What is the Air Products and Chemicals' Operating Profit Margin in 2016?

Subtasks:
Step 1: Retrieve operating income of Air Products and Chemicals in 2015
Step 2: Retrieve net sales of Air Products and Chemicals in 2015
Step 3: Compute operating profit margin

---

Example 3: Two-company calculation
Question: What is the difference of Operating Profit Margin between Air Products and Chemicals in 2016 and International Paper Company in 2006?

Subtasks:
Step 1: Retrieve the operating income of Air Products and Chemicals in 2016  
Step 2: Retrieve net sales of Air Products and Chemicals in 2016
Step 3: Compute operating profit margin of Air Products and Chemicals in 2016
Step 4: Retrieve the operating income of International Paper Company in 2006  
Step 5: Retrieve net sales of International Paper Company in 2006  
Step 6: Compute operating profit margin of International Paper Company in 2006  
Step 7: Compute the difference between the two values from step 3 and 6

---

Now decompose this new question into multiple steps (one subtask per line):

Question: {query}
Subtasks:
"""

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a query decomposition assistant specialized in financial analysis."},
            {"role": "user",   "content": few_shot_prompt}
        ],
        temperature=0.0,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # content = response["choices"][0]["message"]["content"].strip()
    content = response.choices[0].message.content
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return lines

if __name__ == "__main__":
    mcp.run(transport="stdio")
