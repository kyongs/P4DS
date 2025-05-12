import os
import openai
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import json
import re
from tqdm import tqdm
_ = load_dotenv(find_dotenv())

# -------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------
INPUT_FILE = ''     # JSON file containing questions (with explanation and evidence)
OUTPUT_FILE = ''    # JSON file with added 'Level' field for each question

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)

MODEL = 'gpt-4.1-nano'

# -------------------------------------------------------------------------------------
# Prompts
# - system prompt
# - few shot prompt
# -------------------------------------------------------------------------------------
system_prompt = """
You are an expert assistant that classifies financial QA into one of 10 levels:
1: Single-hop information extraction (single-hop, single document)
2: Single-hop numerical calculation (single-hop, single document)  
3: Multi-hop numerical and logical operations (multi-hop, single document)
4: Cross-document operations (multiple documents within DB)  
5: External financial formula/concept search
6: External document lookup (research report, filings)
7: External news-article search
8: Structured user‐provided input (file upload)
9: Unstructured user‐provided input (text/PDF/image upload)
10: Domain expert knowledge required
Respond only with the integer (1–10) corresponding to the level. Do not include any other text.
""".strip()

few_shot_prompt = """\
Classify the following financial QA into Level 1–10 according to the criteria below.

Level 1: Single-hop information extraction (single-hop, single document)
Level 2: Single-hop numerical calculation (single-hop, single document)  
Level 3: Multi-hop numerical and logical operations (multi-hop, single document)
Level 4: Cross-document operations (multiple documents within DB)  
Level 5: External financial formula/concept search
Level 6: External document lookup (research report, filings)
Level 7: External news-article search
Level 8: Structured user‐provided input (file upload)
Level 9: Unstructured user‐provided input (text/PDF/image upload)
Level 10: Domain expert knowledge required

Few-shot examples:

Level 1: What is the FY2018 capital expenditure amount (in USD millions) for 3M?  
Answer: 1

Level 3: Is 3M a capital-intensive business based on FY2022 data? 
Answer: 3

Level 5: What is Amazon’s FY2017 days payable outstanding (DPO)?  
Answer: 5

Level 10: Under IFRS 17, how does the liability for incurred claims change compared to IFRS 4?  
Answer: 10

Now classify the question below:

Question: {question}
{evidence}
{explanation}
Answer:
"""

few_shot_full = """\
Classify the following financial QA into Level 1–10 according to the criteria below.

Level 1: Single-hop information extraction (single-hop, single document)
Level 2: Single-hop numerical calculation (single-hop, single document)  
Level 3: Multi-hop numerical and logical operations (multi-hop, single document)
Level 4: Cross-document operations (multiple documents within DB)  
Level 5: External financial formula/concept search
Level 6: External document lookup (research report, filings)
Level 7: External news-article search
Level 8: Structured user‐provided input (file upload)
Level 9: Unstructured user‐provided input (text/PDF/image upload)
Level 10: Domain expert knowledge required

Few-shot examples:

Level 1: What is the FY2018 capital expenditure amount (in USD millions) for 3M?  
Answer: 1

Level 2: What is the FY2019 fixed asset turnover ratio for Activision Blizzard?
Answer: 2

Level 3: Is 3M a capital-intensive business based on FY2022 data? 
Answer: 3

Level 4: Compare ROA of 3M (FY2018) and Coca Cola (FY2017). Which is higher?
Answer: 4

Level 5: What is Amazon’s FY2017 days payable outstanding (DPO)?  
Answer: 5

Level 6: What was the key agenda of AMCOR’s 8-K filing dated 1 July 2022?
Answer: 6

Level 7: What was the market reaction as reported by Reuters when Company X announced its Q1 earnings?  
Answer: 7

Level 8: Calculate the EBITDA margin from the uploaded Excel sheet for Company X’s segments.
Answer: 8

Level 9: Incorporate the assumptions from my business-plan PDF and compute projected 2025 revenue.
Answer: 9

Level 10: Under IFRS 17, how does the liability for incurred claims change compared to IFRS 4?  
Answer: 10

Now classify the question below:

Question: {question}
{evidence}
{explanation}
Answer:
"""

# -------------------------------------------------------------------------------------
# Functions
# - extract_level: extracts only the integer form of the level
# - classify_level: classifies question level given question, explanation, and evidence
# -------------------------------------------------------------------------------------
def extract_level(answer):
    match = re.search(r'\b([1-9]|10)\b', answer.strip())
    return match.group(1) if match else "Unknown"

def classify_level(question, evidence = None, explanation = None):
    evidence = f"Evidence: {evidence}\n" if evidence else ""
    explanation = f"Explanation: {explanation}\n" if explanation else ""
    user_prompt = few_shot_prompt.format(       # instead, few_shot_full can be used
        question=question,
        evidence=evidence,
        explanation=explanation
    )

    res = client.chat.completions.create(
        model=MODEL,                  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0
    )
    return extract_level(res.choices[0].message.content)

# -------------------------------------------------------------------------------------
# Procedure
# 1. Read file as dataframe
# 2. Classify levels of questions of dataframe
# 3. Save as json file
# -------------------------------------------------------------------------------------
with open(INPUT_FILE) as json_file:
    qa_dict = json.load(json_file)
qa_df = pd.DataFrame(qa_dict)

qa_df['Level'] = 0
for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
    q = row.get('question', row.get('Question'))
    e = row.get('evidence', None)
    x = row.get('explanation', None)
    
    qa_df.at[idx, 'Level'] = classify_level(q, e, x)

qa_json = qa_df.to_json(orient='records', indent=4)
with open(f'OUTPUT_FILE', 'w', encoding='utf-8') as f:
    f.write(qa_json)