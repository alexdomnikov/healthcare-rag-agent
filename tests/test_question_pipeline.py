# Test for retrieve() + rerank() + answer() pipeline

# Usage:
#   uv run python scripts/test_thursday.py

# Verify that:
#   Each doc question produces a cited answer containing [p. X]
#   The out-of-corpus question returns "I don't have that information."
#   Reranker scores print (proves the cross-encoder is running)

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from healthcare_rag.retrieval import retrieve
from healthcare_rag.generation import answer

# Questions
DOC_QUESTIONS = [
    "What are the network adequacy requirements for Medicare Advantage plans?",
    "What does §422.100 require?",
    "What limits exist on how much a Medicare beneficiary can pay out of pocket?",
    "What is the grievance and appeal process for Medicare Advantage?",
    "What marketing materials are Medicare Advantage plans prohibited from using?",
]

OUT_OF_CORPUS = "What is the fair market value of a 2023 Mazda CX-5?"

# Runner
def run():
    print("TEST: retrieve(), rerank(), answer()")

    for i, question in enumerate(DOC_QUESTIONS, 1):
        print(f"[Q{i}] {question}")

        t0 = time.time()
        chunks = retrieve(question, top_k=5, do_rerank=True, mode="hybrid")
        elapsed = time.time() - t0

        if not chunks:
            print("Error: no chunks returned. Check DATABASE_URL and chunk_strategy column\n")
            continue

        print(f"{len(chunks)} chunks, ({elapsed:.1f}s)")
        for j, c in enumerate(chunks[:3], 1):
            page = f"p. {c.page_number}" if c.page_number else "p. ?"
            section = (c.section_path or "")[:55]
            print(f"[{j}] score={c.score:+.3f} {page} {section}")

        t1 = time.time()
        ans = answer(question, chunks)
        print(f"\nAnswer ({time.time()-t1:.1f}s):")
        for line in ans.splitlines():
            print(f"{line}")

        if "[p." not in ans:
            print("Error: no citation found. Check ANSWER_PROMPT refusal instruction")
        else:
            print("Citation present")
        print()

    # Refusal test
    print("REFUSAL TEST (out-of-corpus question)")
    print(f"[Q] {OUT_OF_CORPUS}")
    chunks = retrieve(OUT_OF_CORPUS, top_k=5, do_rerank=True, mode="hybrid")
    ans = answer(OUT_OF_CORPUS, chunks)
    print(f"\nAnswer:")
    for line in ans.splitlines():
        print(f"{line}")

    refused = "don't have that information" in ans.lower()
    print(f"\n{'Correctly refused' if refused else 'Did NOT refuse. Strengthen the refusal instruction in ANSWER_PROMPT'}")

if __name__ == "__main__":
    run()