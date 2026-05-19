from langchain_core.tools import tool

from healthcare_rag.retrieval import retrieve

@tool()
def vector_search(query: str) -> str:
    """Search the CMS Medicare Advantage and Part D Final Rule for regulatory information.

    Use this tool for questions about:
    - Specific regulatory requirements (e.g., network adequacy, MOOP limits, formulary rules)
    - CFR section content (e.g. §422.100, §423.120)
    - Compliance, enrollment, marketing, or grievance procedures

    Do NOT use this tool for:
    - Quantitative questions about specific plans or states (use sql_query instead)
    - Questions about specific drug labels or recalls (use openfda_search instead)
    """
    
    # Vector search tool for Medicare Advantage and Part D final rule pdf search.
    # NOTE: LangChain doc says TYPE HINTS ARE REQUIRED AS THEY DEFINE THE TOOL'S 
    #   INPUT SCHEMA.
    
    chunks = retrieve(query=query, top_k=5)
    if not chunks:
        return "No relevant context found."
    return "\n\n".join(
        f"[p. {c.page_number} | {c.section_path}] {c.text}"
        for c in chunks
    )