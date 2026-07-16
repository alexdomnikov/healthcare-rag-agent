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

JUDGE_MODEL = "qwen/qwen3.6-27b"
RESPONSES_PATH = ROOT / "eval" / "ground_truth_responses.json"
PLACEHOLDER = "[No chunks retrieved — misrouted or tool error.]"

# qwen3.6-27b runs in thinking mode (see _ask) and the faithfulness prompt sends
# the full top-5 context, so a call can spend ~4-5K tokens. Against the free-tier
# 8K TPM cap, two such calls in one window trip a 429. We pace ONE call every 40s
# (de-bursted, not 2/min) so a single large call stays comfortably under the cap;
# spreading calls out avoids the 429 storms that otherwise tank throughput.
# Lower on paid tiers.
JUDGE_CALL_INTERVAL = 40.0

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


# max_retries=0 disables the SDK's silent retry-with-backoff: on a 429 it would
# re-send in the background, inflating token usage and hiding the failure behind
# a long stall. We surface and retry visibly in _ask instead. timeout caps a
# single call so a stuck response read can't hang the whole run.
client = AsyncGroq(max_retries=0, timeout=90.0)

# The limiter caps the token *rate* (TPM); the semaphore caps in-flight
# *concurrency* to one. Firing calls concurrently against the free tier stalled
# on the response read, so we serialize the actual requests while the limiter
# still paces them.
_limiter = AsyncRateLimiter(max_calls=1, period_seconds=JUDGE_CALL_INTERVAL)
_inflight = asyncio.Semaphore(1)


async def _ask(prompt: str, attempts: int = 3) -> float | None:
    # qwen3.6-27b is a reasoning model, and in non-thinking mode it miscalibrates
    # the faithfulness judgment badly (scores well-grounded answers 0). So we keep
    # thinking on and read the verdict from reasoning_format="parsed", which puts
    # the chain-of-thought in a separate field and leaves only the final 0/0.5/1
    # in content. max_tokens is generous so the reasoning finishes before the
    # verdict is emitted — too small a budget truncates content to empty. Retry
    # with backoff on transient errors so a 429 burst doesn't silently score None.
    for attempt in range(attempts):
        await _limiter.acquire()
        start = asyncio.get_event_loop().time()
        try:
            async with _inflight:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0,
                    reasoning_effort="default",
                    reasoning_format="parsed",
                )
            elapsed = asyncio.get_event_loop().time() - start
            text = (r.choices[0].message.content or "").strip()
            match = re.search(r"[01](?:\.5)?", text)
            if match:
                if elapsed > 20:
                    print(f"  slow judge call: {elapsed:.0f}s")
                return float(match.group())
            print(f"judge returned no verdict (attempt {attempt + 1}): {text!r}")
        except Exception as e:
            # Log the exception type + HTTP status only — never the response body,
            # which can echo identifiers (e.g. org id) and adds nothing here.
            status = getattr(e, "status_code", "?")
            elapsed = asyncio.get_event_loop().time() - start
            print(f"judge call failed (attempt {attempt + 1}): "
                  f"{type(e).__name__} {status} after {elapsed:.0f}s")
            await asyncio.sleep(2 ** attempt)
    return None


async def score_question(r: dict) -> dict:
    retrieved = r.get("retrieved_chunks") or [PLACEHOLDER]
    # Faithfulness grades the answer against the context the agent actually
    # conditioned on, so it sees all retrieved chunks (top-5) — otherwise a claim
    # grounded in a rank-4/5 chunk is wrongly marked unsupported. context
    # precision/recall grade retrieval quality, for which the top-3 highest-ranked
    # chunks suffice, and keeping them at 3 holds the run under the free-tier TPD.
    chunks_all = "\n".join(retrieved[:5])
    chunks_top = "\n".join(retrieved[:3])
    q, a, ref = r["question"], r["final_answer"], r["expected_answer"]

    faith, ctx_p, ctx_r = await asyncio.gather(
        _ask(
            f"Does the answer make only claims supported by the context? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nContext:\n{chunks_all}\n\nAnswer:\n{a}"
        ),
        _ask(
            f"Is the context relevant and sufficient to answer the question? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nQuestion:\n{q}\n\nContext:\n{chunks_top}"
        ),
        _ask(
            f"Does the context contain the information needed to produce this reference answer? "
            f"Answer with ONLY 0, 0.5, or 1. No other text.\n\nContext:\n{chunks_top}\n\nReference answer:\n{ref}"
        ),
    )
    print(f"  {r['id']:<6} faith={faith} ctx_p={ctx_p} ctx_r={ctx_r}")
    return {"faithfulness": faith, "context_precision": ctx_p, "context_recall": ctx_r}


async def run_judge(doc_qs: list[dict]) -> list[dict]:
    # Score one question at a time. The limiter + semaphore already serialize the
    # actual API calls, so gathering all 18 questions at once wouldn't add
    # throughput — it would only scramble completion order, so no single question
    # finishes for a long time. Sequential keeps each question's 3 calls grouped
    # and lands verdicts incrementally (observable progress, resumable-in-spirit).
    return [await score_question(r) for r in doc_qs]


# Matches a citation bracket beginning with "p." or "pp." and captures the entire
# body so we can pull every page number out of multi-page brackets like
# "[p. 88, p. 91]" or "[pp. 142-144]". Accepts both ASCII "[...]" and the fullwidth
# "【...】" brackets gpt-oss-120b emits, and a possible space after the opening
# bracket ("[ p. 47 ]"); a stricter pattern silently dropped those and understated
# citation precision.
_CITE_BLOCK = re.compile(r"[\[【]\s*pp?\.?\s*([^\]】]+)[\]】]", re.IGNORECASE)


def _extract_cited_pages(answer: str) -> set[int]:
    cited: set[int] = set()
    for block in _CITE_BLOCK.finditer(answer):
        cited.update(int(n) for n in re.findall(r"\d+", block.group(1)))
    return cited


def citation_precision(answer: str, expected_page) -> float | None:
    # expected_page is an OR: any one of the listed pages is a valid source.
    # We grade only whether the answer's citations land inside that set, not
    # whether the answer reproduces every page in it. Returns None if there's
    # nothing to grade against (e.g. SQL / openFDA / OOS questions).
    cited = _extract_cited_pages(answer)
    expected = to_page_set(expected_page)
    if not expected:
        return None
    if not cited:
        return 0.0
    return round(len(cited & expected) / len(cited), 3)


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
    if not oos:
        return None
    refusals = ["don't have", "do not have", "don't know", "cannot answer"]
    fabricated = [r for r in oos if not any(p in r["final_answer"].lower() for p in refusals)]
    return round(len(fabricated) / len(oos), 3)


def build_markdown(summary: dict, per_q: list[dict], routing: dict) -> str:
    def f(v):
        return f"**{v:.3f}**" if v is not None else "—"

    def g(v):
        return f"{v:.3f}" if v is not None else "—"
    lines = [
        "# Baseline Evaluation",
        f"\n> Judge: `{JUDGE_MODEL}` via Groq (custom LLM-as-judge)\n",
        "## Summary\n",
        "| Metric | Value |", "|--------|-------|",
        f"| Tool routing accuracy    | {f(summary['routing_accuracy'])} ({routing['correct']}/{routing['total']}) |",
        f"| Hallucination rate (OOS) | {f(summary['hallucination_rate'])} |",
        f"| Citation precision (avg) | {f(summary['citation_precision_avg'])} |",
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
        "| ID | Faith | Ctx.P | Ctx.R | CitePrec | Routed |",
        "|----|-------|-------|-------|----------|--------|",
    ]
    for q in per_q:
        ok = "+" if q["routing_correct"] else "x"
        lines.append(f"| {q['id']} | {g(q['faithfulness'])} | {g(q['context_precision'])} "
                     f"| {g(q['context_recall'])} | {g(q['citation_precision'])} | {ok} |")
    return "\n".join(lines) + "\n"


def main():
    responses = json.loads(RESPONSES_PATH.read_text())
    doc_qs = [r for r in responses if r["is_in_corpus"] and r["expected_tool"] == "vector_search"]

    routing = routing_accuracy(responses)
    hall = hallucination_rate(responses)
    for r in doc_qs:
        r["_cite_prec"] = citation_precision(r["final_answer"], r.get("expected_page"))
        r["_routing_correct"] = r["called_tool"] == r["expected_tool"]

    print(f"Routing accuracy : {routing['accuracy']:.3f} ({routing['correct']}/{routing['total']})")
    for m in routing["misrouted"]:
        print(f"x {m['id']} expected={m['expected']} called={m['called']}")
    print(f"Hallucination OOS: {hall}")
    cite_vals = [r["_cite_prec"] for r in doc_qs if r["_cite_prec"] is not None]
    print(f"Citation precision avg : {round(float(np.mean(cite_vals)), 3)}")

    print(f"\nScoring {len(doc_qs)} questions (judge: {JUDGE_MODEL})\n")
    scores = asyncio.run(run_judge(doc_qs))

    per_q = [
        {
            "id": r["id"],
            "faithfulness": s["faithfulness"],
            "context_precision": s["context_precision"],
            "context_recall": s["context_recall"],
            "citation_precision": r["_cite_prec"],
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
        "citation_precision_avg": avg("citation_precision"),
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