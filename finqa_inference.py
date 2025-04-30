#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import re
import openai
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
RATE_LIMIT_SLEEP = 0.1  

# File paths
INPUT_JSON = "dataset/dev.json"
OUTPUT_JSON = "evaluate/predict.json"

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def load_finqa_data(json_path: str) -> list[dict]:
    """
    Load FinQA JSON data from the specified path.
    Returns a list of entries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(item: dict) -> str:
    """
    Combine 'pre_text', 'table', and 'post_text' into a single context string.
    Converts all parts to string to avoid type errors.
    """
    parts = []
    for key in ("pre_text", "table", "post_text"):
        val = item.get(key)
        if val is None:
            continue
        parts.append(str(val))
    return "\n\n".join(parts)


def call_gpt4o_mini(context: str, question: str) -> str:
    """
    Send the context and question to GPT-4o-mini via OpenAI API
    and return the raw DSL program string.
    """
    prompt = (
        "Below is the relevant context from a financial report:\n"
        "==========\n"
        f"{context}\n"
        "==========\n"
        f"Question: {question}\n"
        "Output only the DSL program tokens, ending with EOF."
    )
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_dsl_tokens(raw: str) -> list[str]:
    """
    Parse the raw DSL string into individual tokens.
    Captures function names, numeric refs, numbers, 'none', closing parens, EOF,
    and any intermediate column-name strings.
    """
    pattern = (
        r"(?:subtract\(|add\(|multiply\(|divide\(|"
        r"greater_equal\(|less_equal\(|greater\(|less\(|"
        r"table_average\(|table_sum\(|table_count\(|"
        r"table_max\(|table_min\()|"
        r"#\d+|\d+\.\d+|\d+|none|\)|EOF"
    )
    tokens = []
    last_end = 0
    for m in re.finditer(pattern, raw):
        # extract text between tokens as column names
        if m.start() > last_end:
            txt = raw[last_end:m.start()].strip(", \n")
            if txt:
                tokens.append(txt)
        tokens.append(m.group())
        last_end = m.end()
    return tokens


def batch_inference(data: list[dict]) -> list[dict]:
    """
    Perform batch inference over the dataset.
    Returns predictions list with 'id' and 'predicted' tokens.
    """
    results = []
    total = len(data)
    for idx, item in enumerate(data, start=1):
        doc_id = item.get("id", "<unknown>")
        question = item.get("qa", {}).get("question") or ""
        context = build_context(item)

        print(f"[{idx}/{total}] Processing {doc_id} | Q: {question[:30]}...")
        if not question:
            print("  WARNING: No question found, skipping.")
            continue

        raw_program = call_gpt4o_mini(context, question)
        tokens = parse_dsl_tokens(raw_program)
        results.append({
            "id": doc_id,
            "predicted": tokens
        })
        time.sleep(RATE_LIMIT_SLEEP)

    return results

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if not openai.api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.")
        exit(1)

    print(f"Loading FinQA data from: {INPUT_JSON}")
    data = load_finqa_data(INPUT_JSON)

    print(f"Starting batch inference with model: {MODEL_NAME}")
    predictions = batch_inference(data)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to: {OUTPUT_JSON}")
