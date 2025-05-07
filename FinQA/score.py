from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)

with open('./data/qa_dict.json', 'r') as f:
    qa_dict = json.load(f)

with open('./data/results.json', 'r') as f:
    results = json.load(f)

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
for i, item in enumerate(qa_dict):
  response = results[i]['Output']
  score = score_answer_chain.invoke({'question': item['Question'], 'answer': item['Answer'], 'response': response}).score
  correct += score
  results[i]['Score'] = score

accuracy = correct / len(qa_dict)

print(f"Accuracy: {accuracy}")

with open('./data/results_with_score.json', 'w') as f:
    json.dump(results, f, indent=4)
    print(f"Results with score saved to results_with_score.json")