from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from healthcare_rag.tools.vector_search import vector_search as _vector_search
from healthcare_rag.tools.sql_query import sql_query as _sql_query
from healthcare_rag.tools.openfda_search import openfda_search as _openfda_search

mcp = FastMCP("Healthcare RAG")


@mcp.tool()
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
    return _vector_search.invoke({"query": query})


@mcp.tool()
def sql_query(query: str) -> str:
    """Query the CMS Medicare Advantage 2026 Star Ratings database for quantitative
    questions about plan performance and domain scores.

    Use this tool for questions like:
    - 'How many contracts received a 5-star overall rating?'
    - 'Which contracts have the highest Part C summary rating?'
    - 'What is the average member experience score across all plans?'
    - 'Which parent organizations have the most 4-star-or-above contracts?'
    - 'How do Part C and Part D summary ratings compare on average?'

    Do NOT use this tool for:
    - Questions about regulatory rules or requirements (use vector_search instead)
    - Questions about specific drug labels, adverse events, or recalls (use openfda_search instead)
    """
    return _sql_query.invoke({"question": query})


@mcp.tool()
def openfda_search(drug_name: str, query_type: str = "label") -> str:
    """Search FDA drug data for information about specific medications.

    Use this tool for questions about:
    - Drug labels: indications, warnings, dosage, contraindications, side effects
    - Adverse events reported to the FDA for a specific drug (MedWatch reports)
    - FDA recalls or enforcement actions for a specific drug product

    Parameters:
        drug_name:  The brand or generic name of the drug (e.g. 'Eliquis', 'Ozempic').
        query_type: One of 'label' (default), 'event', or 'enforcement'.

    Do NOT use this tool for:
    - General Medicare Part D formulary or coverage questions (use vector_search)
    - Questions about plan premiums, star ratings, or enrollment counts (use sql_query)
    """
    return _openfda_search.invoke({"drug_name": drug_name, "query_type": query_type})


if __name__ == "__main__":
    mcp.run()
