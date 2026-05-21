import asyncio
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq

from healthcare_rag.eval_metrics import to_page_set

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "docs"

JUDGE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
RESPONSES_PATH = ROOT / "eval" / "ground_truth_responses_v1.json"
PLACEHOLDER = "[No chunks retrieved — misrouted or tool error.]"

# Conservative defaults for Groq's free tier on llama-4-scout. Bump on paid
# tiers — the limiter is what keeps us from getting 429'd, not throughput.
JUDGE_RPM = 30

# Custom llm-as-judge metric calculations. Ragas quickly burned through Groq's limits.
# Usage: uv run eval/llm_eval.py

class AsyncRateLimiter:
    # Sliding-window RPM limiter. Holds the lock only for the timestamp-deque
    # bookkeeping; the actual wait happens lock-free so callers don't serialize.

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0]) + 0.05
            await asyncio.sleep(wait)


client = AsyncGroq()
_limiter = AsyncRateLimiter(max_calls=JUDGE_RPM)


async def _ask(prompt: str) -> float | None:
    # TODO: explicit retry-on-429 with exponential backoff. Today we rely on the
    # groq SDK's default max_retries=2, which is usually enough but swallows the
    # signal — a sustained 429 burst returns None and corrupts the score.
    await _limiter.acquire()
    try:
        r = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        text = r.choices[0].message.content.strip()
        match = re.search(r"[01](?:\.5)?", text)
        return float(match.group()) if match else None
    except Exception as e:
        print(f"judge failed: {e}")
        return None


async def score_question(r: dict) -> dict:
    chunks = "\n".join((r.get("retrieved_chunks") or [PLACEHOLDER])[:3])
    q, a, ref = r["question"], r["final_answer"], r["expected_answer"]

    faith, ctx_p, ctx_r = await asyncio.gather(
        _ask(
            f"Does the answer make only claims supported by the context? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nContext:\n{chunks}\n\nAnswer:\n{a}"
        ),
        _ask(
            f"Is the context relevant and sufficient to answer the question? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nQuestion:\n{q}\n\nContext:\n{chunks}"
        ),
        _ask(
            f"Does the context contain the information needed to produce this reference answer? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nContext:\n{chunks}\n\nReference answer:\n{ref}"
        ),
    )
    print(f"  {r['id']:<6} faith={faith} ctx_p={ctx_p} ctx_r={ctx_r}")
    return {"faithfulness": faith, "context_precision": ctx_p, "context_recall": ctx_r}


async def run_judge(doc_qs: list[dict]) -> list[dict]:
    # Questions run concurrently; the rate limiter paces individual calls so we
    # stay under the per-minute cap without burning wall-clock on sleeps.
    return await asyncio.gather(*(score_question(r) for r in doc_qs))


def citation_f1(answer: str, expected_page) -> float | None:
    cited = {int(p) for p in re.findall(r"\[p\.\s*(\d+)\]", answer)}
    expected = to_page_set(expected_page)
    if not expected or not cited:
        return 0.0 if expected else None
    p = len(cited & expected) / len(cited)
    r = len(cited & expected) / len(expected)
    return round(2 * p * r / (p + r), 3) if (p + r) else 0.0


def routing_accuracy(responses: list[dict]) -> dict:
    misrouted = [r for r in responses if r["called_tool"] != r["expected_tool"]]
    return {
        "accuracy":  round(1 - len(misrouted) / len(responses), 3),
        "correct":   len(responses) - len(misrouted),
        "total":     len(responses),
        "misrouted": [{"id": r["id"], "expected": r["expected_tool"], "called": r["called_tool"]}
                      for r in misrouted],
    }


def hallucination_rate(responses: list[dict]) -> float | None:
    oos = [r for r in responses if not r["is_in_corpus"] and r["expected_tool"] == "none"]
    if not oos: return None
    refusals = ["don't have", "do not have", "don't know", "cannot answer"]
    fabricated = [r for r in oos if not any(p in r["final_answer"].lower() for p in refusals)]
    return round(len(fabricated) / len(oos), 3)


def build_markdown(summary: dict, per_q: list[dict], routing: dict) -> str:
    f = lambda v: f"**{v:.3f}**" if v is not None else "—"
    g = lambda v: f"{v:.3f}" if v is not None else "—"
    lines = [
        "# Baseline Evaluation",
        f"\n> Judge: `{JUDGE_MODEL}` via Groq (custom LLM-as-judge)\n",
        "## Summary\n",
        "| Metric | Value |", "|--------|-------|",
        f"| Tool routing accuracy    | {f(summary['routing_accuracy'])} ({routing['correct']}/{routing['total']}) |",
        f"| Hallucination rate (OOS) | {f(summary['hallucination_rate'])} |",
        f"| Citation F1 (avg)        | {f(summary['citation_f1_avg'])} |",
        f"| Faithfulness             | {f(summary['faithfulness'])} |",
        f"| Context precision        | {f(summary['context_precision'])} |",
        f"| Context recall           | {f(summary['context_recall'])} |",
    ]
    if routing["misrouted"]:
        lines += ["\n## Misrouted\n", "| ID | Expected | Called |", "|----|----------|--------|"]
        for m in routing["misrouted"]:
            lines.append(f"| {m['id']} | {m['expected']} | {m['called']} |")
    lines += [
        "\n## Per-Question\n",
        "| ID | Faith | Ctx.P | Ctx.R | CiteF1 | Routed |",
        "|----|-------|-------|-------|--------|--------|",
    ]
    for q in per_q:
        ok = "+" if q["routing_correct"] else "x"
        lines.append(f"| {q['id']} | {g(q['faithfulness'])} | {g(q['context_precision'])} "
                     f"| {g(q['context_recall'])} | {g(q['citation_f1'])} | {ok} |")
    return "\n".join(lines) + "\n"


def main():
    responses = json.loads(RESPONSES_PATH.read_text())
    doc_qs = [r for r in responses if r["is_in_corpus"] and r["expected_tool"] == "vector_search"]

    routing = routing_accuracy(responses)
    hall = hallucination_rate(responses)
    for r in doc_qs:
        r["_cite_f1"] = citation_f1(r["final_answer"], r.get("expected_page"))
        r["_routing_correct"] = r["called_tool"] == r["expected_tool"]

    print(f"Routing accuracy : {routing['accuracy']:.3f} ({routing['correct']}/{routing['total']})")
    for m in routing["misrouted"]:
        print(f"x {m['id']} expected={m['expected']} called={m['called']}")
    print(f"Hallucination OOS: {hall}")
    cite_vals = [r["_cite_f1"] for r in doc_qs if r["_cite_f1"] is not None]
    print(f"Citation F1 avg : {round(float(np.mean(cite_vals)), 3)}")

    print(f"\nScoring {len(doc_qs)} questions (judge: {JUDGE_MODEL})\n")
    scores = asyncio.run(run_judge(doc_qs))

    per_q = [
        {
            "id": r["id"],
            "faithfulness": s["faithfulness"],
            "context_precision": s["context_precision"],
            "context_recall": s["context_recall"],
            "citation_f1": r["_cite_f1"],
            "routing_correct": r["_routing_correct"],
        }
        for r, s in zip(doc_qs, scores)
    ]

    def avg(key):
        vals = [q[key] for q in per_q if q.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "routing_accuracy": routing["accuracy"],
        "hallucination_rate": hall,
        "citation_f1_avg": avg("citation_f1"),
        "faithfulness": avg("faithfulness"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "routing_detail": routing,
        "per_question": per_q,
    }

    print("\n- Summary --")
    for k, v in summary.items():
        if not isinstance(v, (dict, list)):
            print(f"{k:<28} {v}")

    model_slug = JUDGE_MODEL.replace("/", "-")
    
    DOCS_PATH.mkdir(exist_ok=True)
    (ROOT / "eval" / f"eval_baseline_{model_slug}.json").write_text(json.dumps(summary, indent=2))
    (DOCS_PATH / f"eval_baseline_{model_slug}.md").write_text(build_markdown(summary, per_q, routing))
    print("\nSaved -> root/eval/eval_baseline.json root/docs/eval_baseline.md")

if __name__ == "__main__":
    main()