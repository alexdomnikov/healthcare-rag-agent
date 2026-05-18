# Answer generation takes retrieved chunks, builds a grounded prompt,
#   then calls Qwen3-32B via Groq and returns text.

# I chose Groq over others because there's no meaningful RPM ceiling for a 
#   portfolio demo (~30 RPM free tier), Qwen3-32B has great tool calling support,
#   and it has an OpenAI-compatible API (same interface as everything else in LangChain)

import os
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from healthcare_rag.retrieval import RetrievedChunk

# Prompt
SYSTEM_PROMPT = """\
You are a healthcare regulatory assistant specialising in Medicare Advantage \
and Medicare Part D.

Use ONLY the context passages below to answer the question.
Cite page numbers in square brackets immediately after each claim, like [p. 42].
If multiple passages support a claim, list all relevant pages: [p. 42, p. 67].

If the answer is not in the context, say exactly:
    "I don't have that information."
Do not speculate, use outside knowledge, or invent citations.\
"""

CONTEXT_TEMPLATE = """\
--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}\
"""

# LLM singleton
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    # Return the shared Groq LLM.
    
    # reasoning_effort="none" disables Qwen3's thinking mode. This mode adds 
    #   latency and has known tool-call parsing issues when streaming.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        api_key=api_key,
        reasoning_effort="none",
    )

# Public interface
def answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    # Generates a grounded, cited answer from retrieved chunks.

    # Returns the model's answer as a plain string.
    # If chunks is empty, returns the refusal string directly without an API
    #   call, which saves a round-trip and guarantees correct refusal behaviour.

    if not chunks:
        return "I don't have that information."

    context_parts = []
    for chunk in chunks:
        page_label = f"p. {chunk.page_number}" if chunk.page_number else "p. ?"
        section_part = f" | {chunk.section_path}" if chunk.section_path else ""
        context_parts.append(f"[{page_label}{section_part}]\n{chunk.text}")

    context = "\n\n".join(context_parts)
    user_content = CONTEXT_TEMPLATE.format(context=context, question=question)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = get_llm().invoke(messages)
    return response.content.strip()


def answer_with_metadata(
    question: str,
    chunks: list[RetrievedChunk],
    **kwargs,
) -> dict:
    # Like answer(), but also returns chunk-level data for eval scripts.
    answer_text = answer(question, chunks, **kwargs)
    return {
        "answer": answer_text,
        "retrieved_chunks": [c.text for c in chunks],
        "chunk_pages": [c.page_number for c in chunks],
        "reranker_scores": [c.score for c in chunks],
    }