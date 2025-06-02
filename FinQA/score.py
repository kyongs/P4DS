import argparse
import json
import multiprocessing as mp
import os
from functools import partial

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
    parser.add_argument("--cpu-usage", "-c",
                       type=float, default=0.75,
                       help="CPU usage percentage (0.0-1.0, default: 0.75)")
    return parser.parse_args()


def process_single_item(indexed_data, model_name, temperature, openai_api_key):
    """Process a single question-answer pair for scoring"""
    index, qa_item, result_item = indexed_data
    
    # Initialize LLM for this process
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key)
    
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
    
    try:
        response = result_item['final_answer']
        score = score_answer_chain.invoke({
            'question': qa_item['Question'], 
            'answer': qa_item['Answer'], 
            'response': response
        }).score
        
        # Create processed result item
        processed_result = {
            'Question': qa_item['Question'],
            'System Answer': result_item['final_answer'],
            'True Answer': qa_item['Answer'],
            'Level': qa_item['Level'],
            'Score': score
        }
        
        return index, score, processed_result
        
    except Exception as e:
        print(f"Error processing item {index}: {e}")
        # Return default values for failed items
        processed_result = {
            'Question': qa_item.get('Question', 'ERROR'),
            'System Answer': result_item.get('final_answer', 'ERROR'),
            'True Answer': qa_item.get('Answer', 'ERROR'),
            'Level': qa_item.get('Level', 'ERROR'),
            'Score': 0
        }
        return index, 0, processed_result


def main():
    args = parse_arguments()

    # Load data files
    with open(args.input_qa, 'r') as f:
        qa_dict = json.load(f)

    with open(args.input_results, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(qa_dict)} questions from {args.input_qa}")
    print(f"Loaded {len(results)} results from {args.input_results}")
    
    # Validate data lengths match
    if len(qa_dict) != len(results):
        print(f"Warning: Number of questions ({len(qa_dict)}) does not match number of results ({len(results)})")
        min_len = min(len(qa_dict), len(results))
        print(f"Using first {min_len} items for evaluation")
        qa_dict = qa_dict[:min_len]
        results = results[:min_len]

    # Setup multiprocessing
    num_processes = max(1, int(mp.cpu_count() * args.cpu_usage))
    print(f"Using {num_processes} processes (CPU usage: {args.cpu_usage*100:.1f}%)")
    
    # Create indexed data for order preservation
    indexed_data = [(i, qa_dict[i], results[i]) for i in range(len(qa_dict))]
    
    # Create partial function for multiprocessing
    process_func = partial(
        process_single_item,
        model_name=args.model,
        temperature=args.temperature,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Process items in parallel
    correct = 0
    processed_results = {}
    
    with mp.Pool(processes=num_processes) as pool:
        for index, score, processed_result in tqdm(
            pool.imap(process_func, indexed_data),
            total=len(indexed_data),
            desc="Evaluating answers"
        ):
            correct += score
            processed_results[index] = processed_result
    
    # Sort results by index to maintain original order
    final_results = [processed_results[i] for i in range(len(qa_dict))]
    
    accuracy = correct / len(qa_dict)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(qa_dict)})")

    # Create output directory and save results
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=4)
        print(f"Results with scores saved to {args.output}")


if __name__ == "__main__":
    main()