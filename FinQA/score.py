from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Score FinQA Results")
    parser.add_argument("--input-qa", 
                       default="./data/qa_dict_levels.json",
                       help="Path to input QA dictionary file (default: ./data/qa_dict_levels.json)")
    parser.add_argument("--input-results",
                       default="./results/v0602/results_thoughts_v2.json",
                       help="Path to results file to score (default: ./results/v0602/results_thoughts_v1.json)")
    parser.add_argument("--output",
                       default="./scores/v0602/results_with_score_v2.json",
                       help="Path to output scored results (default: ./scores/v0602/results_with_score_v1.json)")
    parser.add_argument("--model", "-m",
                       default="gpt-4o-mini",
                       help="OpenAI model to use for scoring (default: gpt-4o-mini)")
    parser.add_argument("--temperature", "-t",
                       type=float, default=0.0,
                       help="Model temperature (default: 0.0)")
    return parser.parse_args()

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    args = parse_arguments()
    
    llm = ChatOpenAI(model=args.model, temperature=args.temperature, api_key=OPENAI_API_KEY)

    # Load data files
    with open(args.input_qa, 'r') as f:
        qa_dict = json.load(f)

    with open(args.input_results, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(qa_dict)} questions from {args.input_qa}")
    print(f"Loaded {len(results)} results from {args.input_results}")
    
    # Validate that the number of questions and results match
    if len(qa_dict) != len(results):
        print(f"Warning: Number of questions ({len(qa_dict)}) does not match number of results ({len(results)})")
        min_len = min(len(qa_dict), len(results))
        print(f"Using first {min_len} items for evaluation")
        qa_dict = qa_dict[:min_len]
        results = results[:min_len]

    score_answer_prompt = PromptTemplate(
        input_variables=["question", "answer", "response"],
        template="""
        You have to score the response by comparing with the answer.
        You should score 0 or 1 as JSON with "score" key.
        You can ignore minor difference with the unit or numerical value.
        Question: {question}
        Answer: {answer}
        Response: {response}
        Score:

        Output JSON: {{
          "score": 1 if the response contains the answer, 0 if the response is different from the answer
        }}
        """
    )

    class Score(BaseModel):
        """Score of the response"""
        score: int = Field(description="score of the response")

    score_answer_chain = score_answer_prompt | llm.with_structured_output(Score)

    correct = 0
    for i, item in tqdm(enumerate(qa_dict), desc="Evaluating answers"):
        response = results[i]['final_answer']
        score = score_answer_chain.invoke({'question': item['Question'], 'answer': item['Answer'], 'response': response}).score
        correct += score
        
        # Clean and restructure results
        if 'question' in results[i]:
            del(results[i]['question'])
        if 'trace' in results[i]:
            del(results[i]['trace'])
        
        results[i]['Question'] = item['Question']
        results[i]['System Answer'] = results[i]['final_answer']
        del(results[i]['final_answer'])
        results[i]['True Answer'] = item['Answer']
        results[i]['Level'] = item['Level']
        results[i]['Score'] = score

    accuracy = correct / len(qa_dict)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(qa_dict)})")

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"Results with scores saved to {args.output}")

if __name__ == "__main__":
    main()