import json

import httpx

from healthcare_rag.tools.openfda_search import fetch_openfda

# Sanity check that the openFDA payload is useful for the agent.
# Run: uv run python scripts/smoke_openfda_output.py

if(__name__) == '__main__':
    print(fetch_openfda("Eliquis", "label"))
    print("---")
    print(fetch_openfda("Eliquis", "event"))
    print("---")
    print(fetch_openfda("Lipitor", "enforcement"))
    print("---")
    # Check a nonsense drug name.
    print(fetch_openfda("xyznotadrug_abc999", "label"))

    resp = httpx.get("https://api.fda.gov/drug/event.json", params={
        "search": "Eliquis", 
        "limit": 3
    })
    print(json.dumps(resp.json()["results"][0], indent=2))