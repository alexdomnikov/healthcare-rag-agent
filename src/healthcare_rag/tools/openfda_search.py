import httpx
from langchain_core.tools import tool

BASE_URL = "https://api.fda.gov/drug"
TIMEOUT = 5.0  # seconds (to ensure the agent isn't blocked longer than this)

# Searches openFDA drug data for labels, adverse events, and recalls.
# Endpoints used:
#   /drug/label.json (prescribing info, indications, warnings, dosage)
#   /drug/event.json (MedWatch adverse event reports)
#   /drug/enforcement.json (recalls and enforcement actions)

# Summariser helpers to keep each result type readable in one pass
def _summarize_label(results: list[dict]) -> str:
    # Pull the most useful fields from a drug-label result.
    lines = []
    # Show at most 1 label record
    for r in results[:1]:
        openfda = r.get("openfda", {})
        brand = ", ".join(openfda.get("brand_name", ["unknown brand"]))
        generic = ", ".join(openfda.get("generic_name", ["unknown generic"]))
        lines.append(f"Drug: {brand} ({generic})")

        for field in ("indications_and_usage", "warnings", "dosage_and_administration",
                      "contraindications", "adverse_reactions"):
            text = r.get(field)
            if text:
                # FDA label fields are lists of strings
                snippet = text[0][:600] if isinstance(text, list) else text[:600]
                label = field.replace("_", " ").title()
                lines.append(f"  {label}: {snippet}...")
        # Blank line between records
        lines.append("")
    return "\n".join(lines).strip()

def _summarize_event(results: list[dict]) -> str:
    # Summarise adverse-event reports (show patient outcomes and top reactions)."""
    lines = []
    for i, r in enumerate(results[:3], 1):
        patient = r.get("patient", {})
        reactions = patient.get("reaction", [])
        outcomes = r.get("patientoutcome", [])
        reactions_text = ", ".join(
            rx.get("reactionmeddrapt", "unknown") for rx in reactions[:5]
        )
        outcome_map = {
            "1": "recovered", "2": "recovering", "3": "not recovered",
            "4": "recovered with sequelae", "5": "fatal", "6": "unknown",
        }
        outcomes_text = ", ".join(
            outcome_map.get(str(o), "unknown") for o in outcomes
        )
        report_date = r.get("receiptdate", "unknown date")
        serious = r.get("serious", "unknown")
        lines.append(
            f"Report {i} ({report_date}) — Serious: {serious}\n"
            f"Reactions: {reactions_text}\n"
            f"Outcomes: {outcomes_text}"
        )
    return "\n".join(lines)

def _summarize_enforcement(results: list[dict]) -> str:
    # Summarise recall / enforcement records.
    lines = []
    for r in results[:3]:
        lines.append(
            f"Recall class: {r.get('classification', 'N/A')} | "
            f"Status: {r.get('status', 'N/A')} | "
            f"Date: {r.get('recall_initiation_date', 'N/A')}\n"
            f"Product: {r.get('product_description', 'N/A')[:200]}\n"
            f"Reason: {r.get('reason_for_recall', 'N/A')[:300]}"
        )
    return "\n".join(lines)

SUMMARIZERS = {
    "label": _summarize_label,
    "event": _summarize_event,
    "enforcement": _summarize_enforcement,
}

# Core fetch (kept separate so it's easy to unit test without LangChain)
def fetch_openfda(drug_name: str, query_type: str) -> str:
    # Hit the openFDA API and return a formatted string (or an error message).
    # NOTE: Never raises; errors returned as strings so agents can reason about them.

    if query_type not in SUMMARIZERS:
        return (
            f"Invalid query_type '{query_type}'. "
            "Choose from: 'label', 'event', or 'enforcement'."
        )

    url    = f"{BASE_URL}/{query_type}.json"
    if query_type == "event":
        search = f'openfda.brand_name:"{drug_name}"+AND+drugcharacterization:"1"'
    else:
        search = f'openfda.brand_name:"{drug_name}"'

    params = {
        "search": search,
        "limit": 3,
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(url, params=params)

        # If brand search returned nothing, fall back to a broader text search
        if resp.status_code == 404 or (
            resp.status_code == 200
            and not resp.json().get("results")
        ):
            if query_type == "event":
                params["search"] = f'{drug_name}+AND+drugcharacterization:"1"'
            else:
                params["search"] = drug_name
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.get(url, params=params)

        resp.raise_for_status()

    except httpx.TimeoutException:
        return (
            f"openFDA request timed out after {TIMEOUT}s for '{drug_name}' "
            f"({query_type}). The FDA API may be slow. Try again or rephrase."
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return (
                f"No openFDA {query_type} records found for '{drug_name}'. "
                "Check the spelling or try the generic name."
            )
        return f"openFDA HTTP error {exc.response.status_code}: {exc}"
    except httpx.RequestError as exc:
        return f"openFDA network error: {exc}"

    data    = resp.json()
    results = data.get("results", [])
    if not results:
        return (
            f"No openFDA {query_type} records found for '{drug_name}'. "
            "Try the generic name or check spelling."
        )

    total = data.get("meta", {}).get("results", {}).get("total", "unknown")
    header = (
        f"openFDA {query_type} results for '{drug_name}' "
        f"(showing {len(results)} of {total} total records):\n\n"
    )
    return header + SUMMARIZERS[query_type](results)

# Agent's tool
@tool("openfda_search")
def openfda_search_tool(drug_name: str, query_type: str = "label") -> str:
    """Search FDA drug data for information about specific medications.

    Use this tool for questions about:
    - Drug labels: indications, warnings, dosage, contraindications, side effects
    - Adverse events reported to the FDA for a specific drug (MedWatch reports)
    - FDA recalls or enforcement actions for a specific drug product

    Parameters:
        drug_name:  The brand or generic name of the drug. Examples: 'Eliquis',
                    'apixaban', 'Ozempic', 'semaglutide', 'Lipitor', 'atorvastatin'.
        query_type: One of:
                    'label' — prescribing information (default)
                    'event' — adverse event reports
                    'enforcement' — recalls and FDA enforcement actions

    Do NOT use this tool for:
    - General Medicare Part D formulary or coverage questions (use vector_search)
    - Questions about plan premiums, star ratings, or enrollment counts (use sql_query)
    - Regulatory rules about how Part D formularies must be structured (use vector_search)
    """
    return fetch_openfda(drug_name=drug_name, query_type=query_type)