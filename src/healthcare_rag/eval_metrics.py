# Retrieval-evaluation helpers shared by eval/metrics.py and the ablation
# runners. Lives in the installed package so every eval script can import it
# without sys.path tricks.

ExpectedPage = int | list[int] | None


def to_page_set(expected_page: ExpectedPage) -> set[int]:
    if expected_page is None:
        return set()
    if isinstance(expected_page, list):
        return {int(p) for p in expected_page if p is not None}
    return {int(expected_page)}


def recall_at_k(retrieved_pages: list[int | None], expected_page: ExpectedPage, k: int) -> int:
    pages = to_page_set(expected_page)
    if not pages:
        return 0
    return int(bool(pages & set(retrieved_pages[:k])))


def reciprocal_rank(retrieved_pages: list[int | None], expected_page: ExpectedPage) -> float:
    pages = to_page_set(expected_page)
    if not pages:
        return 0.0
    for i, page in enumerate(retrieved_pages):
        if page in pages:
            return 1.0 / (i + 1)
    return 0.0
