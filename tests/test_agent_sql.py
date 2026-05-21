import json
from dotenv import load_dotenv
from pathlib import Path

from healthcare_rag.core import get_agent

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / 'eval'

# Tests agent's SQL search tool for CMS Star Ratings csv data.
# Run: uv run tests/test_agent_sql.py

with open(EVAL_PATH / "ground_truth.json") as f:
    ground_truth = json.load(f)

sql_questions = [
    q for q in ground_truth
    if q["expected_tool"] == "sql_query"
]

agent = get_agent()

for q in sql_questions:
    print(f"\nQ: {q['question']}")
    result = agent.invoke({"messages": [{"role": "user", "content": q["question"]}]})
    print(f"A: {result['messages'][-1].content}")