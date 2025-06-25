import json
import os
import re
import sys
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are an expert financial QA explainer. Given a ReAct trace, explain-"
    "step by step-*why* the agent produced its final answer. Quote and cite the "
    "exact document lines (file path + page) that justify each step, then end "
    "with a one-sentence summary restating the final answer. Use clear, "
    "reader-friendly English and bullet points.\n\n"
    "At the end, include a separate section titled 'Citation:' listing all cited sources "
    "in the format: <file name - page>."
)

OBS_RE = re.compile(r"^([\w/.-]+\.pdf-\d+)\s*:\s*(.*)$")


def parse_trace(trace: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for line in trace.splitlines() if (m := OBS_RE.match(line.strip()))]


def extract_paragraphs(trace: str, cited: List[str]) -> str:
    # lines = trace.splitlines()
    # trace가 딕셔너리인 경우 처리
    if isinstance(trace, dict):
        trace_str = trace.get("trace_log", "")
    else:
        trace_str = trace
    
    lines = trace_str.splitlines()
    paragraph_blocks = {}
    current_citation = None
    buffer = []

    for line in lines:
        match = OBS_RE.match(line.strip())
        if match:
            if current_citation and current_citation in cited:
                paragraph_blocks.setdefault(current_citation, []).extend(buffer)
            current_citation = match.group(1).strip()
            buffer = [match.group(2).strip()]
        elif current_citation:
            buffer.append(line.strip())

    if current_citation and current_citation in cited:
        paragraph_blocks.setdefault(current_citation, []).extend(buffer)

    return "\n---\n".join([
        f"{src}:\n" + "\n".join(lines).strip()
        for src, lines in paragraph_blocks.items()
    ])


# def build_user_prompt(rec: Dict) -> str:
#     obs_block = "\n".join(f"{src}: {para}" for src, para in parse_trace(rec["trace"]))
#     return (
#         f"### Question\n{rec['question']}\n\n"
#         f"### Final Answer (agent)\n{rec['final_answer']}\n\n"
#         f"### Agent Trace\n{rec['trace']}\n\n"
#         f"### Parsed Observations\n{obs_block}"
#     )

def build_user_prompt(rec: Dict) -> str:
    # trace가 딕셔너리인 경우 처리
    trace_data = rec["trace"]
    if isinstance(trace_data, dict):
        trace_str = trace_data.get("trace_log", "")
    else:
        trace_str = trace_data
    
    obs_block = "\n".join(f"{src}: {para}" for src, para in parse_trace(trace_str))
    return (
        f"### Question\n{rec['question']}\n\n"
        f"### Final Answer (agent)\n{rec['final_answer']}\n\n"
        f"### Agent Trace\n{trace_str}\n\n"
        f"### Parsed Observations\n{obs_block}"
    )


def explain_sync(index: int, rec: Dict, score: Dict, api_key: str) -> Tuple[int, Dict]:
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(rec)},
            ],
        )
        full_output = resp.choices[0].message.content.strip()

        citation_match = re.search(r"Citation:\s*(.*)$", full_output, re.DOTALL | re.IGNORECASE)
        citation = citation_match.group(1).strip() if citation_match else ""
        explanation = full_output.replace(citation_match.group(0), "").strip() if citation_match else full_output

        # parse cited sources
        # cited_sources = re.findall(r"([\w/.-]+\.pdf-\d+)", citation)
        # citation_paragraph = extract_paragraphs(rec["trace"], cited_sources)

        # parse cited sources
        cited_sources = re.findall(r"([\w/.-]+\.pdf-\d+)", citation)
        
        # trace가 딕셔너리인 경우 처리
        trace_data = rec["trace"]
        if isinstance(trace_data, dict):
            trace_for_extraction = trace_data.get("trace_log", "")
        else:
            trace_for_extraction = trace_data
        
        citation_paragraph = extract_paragraphs(trace_for_extraction, cited_sources)

    except Exception as e:
        explanation, citation, citation_paragraph = f"[ERROR] {e}", "", ""

    return index, {
        "question": rec["question"],
        "final_answer": rec["final_answer"],
        "true_answer": score["True Answer"],
        "score": score["Score"],
        "explain": explanation,
        "citation": citation,
        "citation_paragraph": citation_paragraph,
    }


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    input_path = "./results/v0608/results_thoughts_v8.json"
    score_path = "./scores/v0608/results_with_score_v8.json"
    output_path = "explanations.json"
    api_key = os.getenv("OPENAI_API_KEY")

    records = load_json(input_path)
    scores = load_json(score_path)

    num_proc = max(1, int(mp.cpu_count() * 0.7))
    print(f"Using {num_proc} processes for explanation...")

    indexed_data = [(i, rec, score, api_key) for i, (rec, score) in enumerate(zip(records, scores))]

    with mp.Pool(processes=num_proc) as pool:
        results = pool.starmap(explain_sync, indexed_data)

    results_sorted = [r for _, r in sorted(results, key=lambda x: x[0])]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path} ({len(results_sorted)} records)")


if __name__ == "__main__":
    main()
