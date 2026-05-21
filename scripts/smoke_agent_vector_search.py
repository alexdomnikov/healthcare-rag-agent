import json
from dotenv import load_dotenv
from pathlib import Path

from healthcare_rag.core import get_agent
from healthcare_rag.retrieval import retrieve

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / 'eval'

# Integration smoke script: invokes the live agent on a sample of doc questions
# and prints each Q&A pair alongside the expected answer for manual inspection.
# No assertions; rigorous scoring lives in eval/.
# Run: uv run python scripts/smoke_agent_vector_search.py

# Warm up: forces model loading before any questions run
retrieve("Medicare Part D", top_k=1)

with open(EVAL_PATH / "ground_truth.json") as f:
    ground_truth = json.load(f)

doc_questions = [
    q for q in ground_truth 
    if q["expected_tool"] == "vector_search" and q["is_in_corpus"]
]

agent = get_agent()
# Just taking a quick peek at the first 5 ground truth Q&A pairs.
for q in doc_questions[:5]:
    print(f"\nQ: {q['question']}")
    result = agent.invoke({"messages": [{"role": "user", "content": q["question"]}]})
    print(f"A: {result['messages'][-1].content}")
    # NOTE: this isn't a rigorous scoring or anything, just a quick sanity check.
    print(f"Expected: {q['expected_answer']}")
    print(f"Expected page: {q['expected_page']}")