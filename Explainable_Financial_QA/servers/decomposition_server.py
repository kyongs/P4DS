# servers/decomposition_server.py

from typing import Annotated, Dict, List, Any
from mcp.server.fastmcp import FastMCP
import openai
import os
import json
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

mcp = FastMCP("Decompose")

@mcp.tool(description="Analyze a financial query and determine if decomposition is needed. If so, break it down into dependency-aware subtasks.")
def decompose_query(
    query: Annotated[str, "The financial question to analyze and potentially break into subtasks."]
) -> Dict[str, Any]:
    """
    Analyzes a financial query to determine if decomposition is necessary.
    Returns either a simple execution plan or a dependency-aware task breakdown.
    Uses VERY CONSERVATIVE decomposition criteria - only decomposes specific patterns.
    """
    
    # Check if OpenAI API key is available
    openai_api_key = api_key
    
    if openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
    
    # OpenAI-based implementation with VERY CONSERVATIVE decomposition criteria
    analysis_prompt = r"""
You are an expert at analyzing financial queries with a VERY CONSERVATIVE approach to decomposition.

IMPORTANT: You must respond with ONLY valid JSON. Do not include any text before or after the JSON response.

CRITICAL: Use EXTREMELY CONSERVATIVE criteria for decomposition. Almost all queries should be executed directly without decomposition.

DECOMPOSE ONLY these specific patterns:
1. Questions that require company identification by characteristics (sector, founding year, etc.) followed by multi-year data comparison
   Example: "What is the difference in [financial metric] for the [sector] company founded in [year] between [year1] and [year2]?"

KEEP SIMPLE: All other questions including:
- Direct company name lookups
- Single year queries
- Simple ratio calculations
- Any questions with explicit company names
- Operating profit margin calculations
- Current ratio when company is explicitly named

RESPONSE FORMAT (JSON ONLY):
{{
  "needs_decomposition": boolean,
  "complexity_reason": "brief explanation",
  "execution_plan": {{
    "type": "simple" | "parallel" | "sequential" | "mixed",
    "tasks": [
      {{
        "id": "task_1",
        "description": "what this task does",
        "depends_on": [],
        "execution_group": 1
      }}
    ]
  }}
}}

EXAMPLES:

Company identification + multi-year comparison (DECOMPOSE):
{{
  "needs_decomposition": true,
  "complexity_reason": "Requires company identification by characteristics followed by multi-year data comparison",
  "execution_plan": {{
    "type": "sequential",
    "tasks": [
      {{"id": "task_1", "description": "Use retrieve_factual_data to identify the financial sector company founded in 1869", "depends_on": [], "execution_group": 1}},
      {{"id": "task_2", "description": "Use retrieve_factual_data to get the identified company's total assets for 2013", "depends_on": ["task_1"], "execution_group": 2}},
      {{"id": "task_3", "description": "Use retrieve_factual_data to get the identified company's total assets for 2017", "depends_on": ["task_1"], "execution_group": 2}},
      {{"id": "task_4", "description": "Calculate the difference between 2017 and 2013 total assets", "depends_on": ["task_2", "task_3"], "execution_group": 3}}
    ]
  }}
}}

Direct company lookup (SIMPLE):
{{
  "needs_decomposition": false,
  "complexity_reason": "Direct company lookup with explicit name",
  "execution_plan": {{
    "type": "simple",
    "tasks": [{{"id": "task_1", "description": "Use retrieve_factual_data to get Apple's revenue for 2020", "depends_on": [], "execution_group": 1}}]
  }}
}}

Single year query (SIMPLE):
{{
  "needs_decomposition": false,
  "complexity_reason": "Single year query can be handled in one retrieval",
  "execution_plan": {{
    "type": "simple",
    "tasks": [{{"id": "task_1", "description": "Use retrieve_factual_data to get Microsoft's operating income for 2019", "depends_on": [], "execution_group": 1}}]
  }}
}}

All others (SIMPLE):
{{
  "needs_decomposition": false,
  "complexity_reason": "Can be handled in single retrieval",
  "execution_plan": {{
    "type": "simple",
    "tasks": [{{"id": "task_1", "description": "Use retrieve_factual_data to retrieve required data", "depends_on": [], "execution_group": 1}}]
  }}
}}

Now analyze this query with VERY CONSERVATIVE decomposition criteria and return ONLY the JSON response:
Query: {query}
"""

    try:
        print(f"[DEBUG] Analyzing query with CONSERVATIVE OpenAI criteria: {query}")
        
        # Initialize OpenAI client with proper v1.0+ API
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Call the OpenAI ChatCompletion endpoint using new API format
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a query analysis assistant with EXTREMELY CONSERVATIVE decomposition criteria. Only decompose queries that require company identification by characteristics followed by multi-year data comparison. You must respond with ONLY valid JSON format - no explanations, no markdown, no additional text."},
                {"role": "user", "content": analysis_prompt.format(query=query)}
            ],
            temperature=0.0,
            max_tokens=800,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        content = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI response: {content}")
        
        # Parse the JSON response
        try:
            # Clean up the response in case there are markdown code blocks
            cleaned_content = content
            
            # Remove markdown code blocks
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing whitespace and newlines
            cleaned_content = cleaned_content.strip()
            
            # Try to find JSON object boundaries if the content is malformed
            if not cleaned_content.startswith('{'):
                # Look for the first opening brace
                start_idx = cleaned_content.find('{')
                if start_idx != -1:
                    cleaned_content = cleaned_content[start_idx:]
            
            if not cleaned_content.endswith('}'):
                # Look for the last closing brace
                end_idx = cleaned_content.rfind('}')
                if end_idx != -1:
                    cleaned_content = cleaned_content[:end_idx + 1]
            
            print(f"[DEBUG] Cleaned content for parsing: {cleaned_content[:200]}...")
            
            result = json.loads(cleaned_content)
            
            # Additional validation for specific decomposition pattern
            if result.get("needs_decomposition", False):
                task_count = len(result.get("execution_plan", {}).get("tasks", []))
                
                # Check for the specific pattern: company identification + multi-year comparison
                pattern_keywords = ["difference", "between", "founded", "sector", "company"]
                has_pattern = any(keyword in query.lower() for keyword in pattern_keywords)
                
                if has_pattern and "difference" in query.lower() and "between" in query.lower():
                    print(f"[DEBUG] Decomposition approved for company identification + multi-year pattern: {query}")
                else:
                    print(f"[DEBUG] Decomposition rejected - pattern not matching: {query}")
                    # Override to simple execution
                    result = {
                        "needs_decomposition": False,
                        "complexity_reason": "Pattern does not match company identification + multi-year comparison requirement",
                        "execution_plan": {
                            "type": "simple",
                            "tasks": [
                                {
                                    "id": "task_1",
                                    "description": query,
                                    "depends_on": [],
                                    "execution_group": 1
                                }
                            ]
                        }
                    }
            
            print(f"[DEBUG] Final decomposition result: {result}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON response: {e}")
            print(f"[ERROR] Raw content: {content}")
            print(f"[ERROR] Cleaned content: {cleaned_content[:500]}")
            # Return simple fallback
            return {
                "needs_decomposition": False,
                "complexity_reason": f"Failed to parse OpenAI JSON response: {str(e)}",
                "execution_plan": {
                    "type": "simple",
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": query,
                            "depends_on": [],
                            "execution_group": 1
                        }
                    ]
                }
            }
        
    except Exception as e:
        print(f"[ERROR] OpenAI decomposition analysis failed: {e}")
        # Return simple fallback - conservative approach
        return {
            "needs_decomposition": False,
            "complexity_reason": f"OpenAI API error - defaulting to simple execution: {e}",
            "execution_plan": {
                "type": "simple",
                "tasks": [
                    {
                        "id": "task_1",
                        "description": query,
                        "depends_on": [],
                        "execution_group": 1
                    }
                ]
            }
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")