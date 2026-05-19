from langchain.agents import create_agent
from healthcare_rag.generation import get_llm
from healthcare_rag.tools.vector_search import vect_search_tool

SYSTEM_PROMPT = """You are a healthcare regulatory assistant. Answer questions 
using the available tools. Always cite page numbers in brackets like [p. 142] 
when answering from regulatory text. If multiple page numbers support your 
answer, then list all of them. DO NOT ANSWER BEFORE CALLING A TOOL - THE MOST RELEVANT
TOOL ACCORDING TO TOOL DESCRIPTIONS. If the selected tool returns no relevant context, 
then say exactly "I don't have that information." Do not speculate, use outside knowledge, 
or invent citations. Always use specified tools to answer questions. Do not 
answer from memory.
"""

# get_llm() is cached, so calling get_agent() multiple times won't spin up new
#    LLM instances. Only the agent wrapper gets recreated.
def get_agent():
    return create_agent(
        model=get_llm(),
        tools=[vect_search_tool],
        system_prompt=SYSTEM_PROMPT,
    )