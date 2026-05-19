import re
from sqlalchemy import text
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from healthcare_rag.core import get_llm, get_readonly_engine

# SQL generation prompt. This tool is used to search CMS Star Ratings csv data.
SQL_SYSTEM_PROMPT = """
You generate SQL SELECT queries for the CMS Medicare Advantage 2026 Star Ratings database.

Available tables:

  cms_summary_ratings(
      contract_number, -- e.g. 'H0028', 'E3014'
      organization_type, -- e.g. 'Local CCP', 'PFFS', 'Employer/Union Only Direct Contract PDP'
      contract_name,
      organization_marketing_name,
      parent_organization,
      snp, -- 'Yes' or 'No'
      disaster_2023, -- numeric % of disaster-affected enrollees
      disaster_2024,
      part_c_summary_2026, -- NUMERIC 1.0-5.0, NULL if not applicable
      part_d_summary_2026, -- NUMERIC 1.0-5.0, NULL if not applicable
      overall_2026 -- NUMERIC 1.0-5.0, NULL if not applicable
  )

  cms_domain_stars(
      contract_number,
      organization_type,
      contract_name,
      organization_marketing_name,
      parent_organization,
      hd1_staying_healthy_screenings_tests_and_vaccines, -- NUMERIC 1-5 or NULL
      hd2_managing_chronic_long_term_conditions, -- NUMERIC 1-5 or NULL
      hd3_member_experience_with_health_plan, -- NUMERIC 1-5 or NULL
      hd4_member_complaints_and_changes_in_the_health_plan_s_performance, -- NUMERIC 1-5 or NULL
      hd5_health_plan_customer_service, -- NUMERIC 1-5 or NULL
      dd1_drug_plan_customer_service, -- NUMERIC 1-5 or NULL
      dd2_member_complaints_and_changes_in_the_drug_plan_s_performance, -- NUMERIC 1-5 or NULL
      dd3_member_experience_with_the_drug_plan, -- NUMERIC 1-5 or NULL
      dd4_drug_safety_and_accuracy_of_drug_pricing -- NUMERIC 1-5 or NULL
  )

Rules:
  1. SELECT only. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
  2. Always include LIMIT (default LIMIT 5).
  3. These tables have no state column. Data is by contract, not geography.
  4. Rating columns are NUMERIC with NULLs for non-applicable contracts.
     Always filter with WHERE <col> IS NOT NULL before aggregating.
  5. contract_number uses CMS H-numbers like 'H1234'. Never filter by integer.
  6. The two tables join on contract_number.
  7. When filtering for "4 stars or above", "4 or higher", "at least 4", etc.,
     always use >= not =. Always combine with IS NOT NULL: e.g.,
     WHERE overall_2026 >= 4 AND overall_2026 IS NOT NULL
""".strip()

_FORBIDDEN = [
    "insert", "update", "delete", "drop",
    "alter", "create", "truncate", "grant", "revoke",
]

# Structured output schema
class SQLQuery(BaseModel):
    sql: str = Field(description="A read-only SELECT query. No DDL. No DML. Must include LIMIT.")
    explanation: str = Field(description="One sentence explaining what the query computes.")

# Safety guard
def _validate(sql: str) -> str | None:
    # Returns an error string if unsafe, None if clean.
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        return "Query does not start with SELECT."
    for kw in _FORBIDDEN:
        if re.search(rf"\b{kw}\b", lowered):
            return f"Query contains forbidden keyword: '{kw}'."
    return None

@tool()
def sql_query(question: str) -> str:
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
    # Generate SQL via structured output
    structured_llm = get_llm().with_structured_output(SQLQuery)
    try:
        query_obj: SQLQuery = structured_llm.invoke([
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ])
    except Exception as e:
        return f"Failed to generate SQL: {e}"

    # Validate
    error = _validate(query_obj.sql)
    if error:
        return f"Query rejected: {error}\nGenerated SQL was:\n{query_obj.sql}"

    # Execute as read-only
    try:
        with get_readonly_engine().connect() as conn:
            result = conn.execute(text(query_obj.sql))
            rows = result.fetchall()
            col_names = list(result.keys())
    except Exception as e:
        return (
            f"Query execution failed: {e}\n"
            f"SQL: {query_obj.sql}\n"
            f"Explanation: {query_obj.explanation}"
        )

    if not rows:
        return (
            f"Query returned no results.\n"
            f"SQL: {query_obj.sql}\n"
            f"Explanation: {query_obj.explanation}"
        )

    # Format as a readable table
    header = " | ".join(col_names)
    divider = "-" * len(header)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)

    return (
        f"Explanation: {query_obj.explanation}\n"
        f"SQL: {query_obj.sql}\n\n"
        f"{header}\n{divider}\n{body}\n\n"
        f"({len(rows)} row(s) returned)"
    )
