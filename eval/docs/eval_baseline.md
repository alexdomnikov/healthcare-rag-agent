# Baseline Evaluation

> Judge: `llama-3.3-70b-versatile` via Groq

## Summary

| Metric | Value |
|--------|-------|
| Tool routing accuracy    | **0.929** (26/28) |
| Hallucination rate (OOS) | **0.000** |
| Citation F1 (avg)        | **0.245** |
| Faithfulness             | **1.000** |
| Context precision        | **0.920** |
| Context recall           | **1.000** |

## Misrouted

| ID | Expected | Called |
|----|----------|--------|
| q013 | vector_search | sql_query |
| q014 | vector_search | none |

## Per-Question

| ID | Faith | Ctx.P | Ctx.R | CiteF1 | Routed |
|----|-------|-------|-------|--------|--------|
| q006 | 1.000 | 1.000 | 1.000 | 0.667 | + |
| q004 | — | — | — | 0.000 | + |
| q010 | 1.000 | 1.000 | 1.000 | 0.667 | + |
| q002 | — | — | — | 1.000 | + |
| q009 | 1.000 | 1.000 | 1.000 | 0.000 | + |
| q011 | 1.000 | 0.679 | 1.000 | 0.400 | + |
| q012 | — | — | — | 0.400 | + |
| q016 | — | — | — | 0.000 | + |
| q003 | — | — | — | 0.364 | + |
| q008 | — | — | — | 0.000 | + |
| q015 | — | — | — | 0.000 | + |
| q005 | — | — | — | 0.667 | + |
| q017 | — | — | — | 0.000 | + |
| q001 | — | — | — | 0.000 | + |
| q018 | — | — | — | 0.000 | + |
| q013 | — | — | — | 0.000 | x |
| q014 | — | — | — | 0.000 | x |
