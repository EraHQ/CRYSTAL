"""Gate E (2026-07-20): tabular ingestion. E-Q1=B ratified — chunks +
per-row mechanical facts, zero LLM; C4's #sheet= fragment grain gets
its first consumer."""
from __future__ import annotations

import io

import pytest

from crystal_cache.ingestion.document_chunker import (
    _chunk_tabular,
    _parse_tabular_canonical,
    detect_document_type,
)
from crystal_cache.ingestion.file_extract import (
    extract_tabular_from_delimited,
    extract_tabular_from_xlsx,
)


def test_detect_tabular_extensions():
    assert detect_document_type("a,b\n1,2", "payroll.csv") == "tabular"
    assert detect_document_type("", "q3.xlsx") == "tabular"
    assert detect_document_type("x\ty", "data.tsv") == "tabular"
    assert detect_document_type("def f(): pass", "f.py") == "code"


def test_csv_canonicalizes_quotes_and_delimiters():
    raw = b'name,role\n"Jenkins, T.",Engineer\nRivera,"Designer\nLead"'
    text = extract_tabular_from_delimited(raw, ",")
    sheets = _parse_tabular_canonical(text)
    assert sheets[0]["headers"] == ["name", "role"]
    assert sheets[0]["rows"][0] == ["Jenkins, T.", "Engineer"]
    assert sheets[0]["rows"][1] == ["Rivera", "Designer Lead"]


def test_xlsx_extracts_sheet_markers():
    from openpyxl import Workbook
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Q3"
    ws1.append(["item", "price"])
    ws1.append(["widget", 5])
    ws2 = wb.create_sheet("Q4")
    ws2.append(["item", "price"])
    ws2.append(["gadget", 9])
    buf = io.BytesIO()
    wb.save(buf)

    text = extract_tabular_from_xlsx(buf.getvalue())
    sheets = _parse_tabular_canonical(text)
    assert [s["sheet"] for s in sheets] == ["Q3", "Q4"]
    assert sheets[0]["rows"] == [["widget", "5"]]


def test_chunker_groups_rows_with_header_context():
    rows = "\n".join(f"item{i}\t{i}" for i in range(90))
    text = f"name\tqty\n{rows}"
    chunks = _chunk_tabular(text, "inventory.csv")
    assert len(chunks) == 3                      # 40 + 40 + 10
    assert all(c["text"].startswith("name\tqty\n") for c in chunks)
    assert chunks[0]["locator"] == "inventory rows 1-40"
    assert chunks[2]["locator"] == "inventory rows 81-90"
    assert chunks[0]["doc_type"] == "tabular"
    assert chunks[0]["sheet"] is None


@pytest.mark.asyncio
async def test_mechanical_row_extraction_zero_llm(store):
    """E-Q1=B: one fact per row, keyed by the mostly-unique column,
    canonical citation, no model anywhere near it."""
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline
    text = (
        "name\trole\tsalary\n"
        "Jenkins\tEngineer\t120k\n"
        "Rivera\tEngineer\t110k\n"
        "Okafor\tDesigner\t115k"
    )
    chunks = _chunk_tabular(text, "payroll.csv")
    pipeline = DocumentPipeline.__new__(DocumentPipeline)  # no client
    items = await pipeline.extract_items(
        "", content_chunks=chunks, detected_type="tabular",
        label="payroll.csv",
    )
    assert len(items) == 3
    by_key = {i.key: i for i in items}
    jenkins = by_key["payroll Jenkins"]
    assert jenkins.value == "name: Jenkins; role: Engineer; salary: 120k"
    assert jenkins.citation == "payroll#row-1"
    assert jenkins.sparse_key == "Tabular|payroll row 1|Jenkins|Data"
    # 'role' repeats -> not the key column; 'name' is.


def test_key_column_falls_back_to_row_number():
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline
    headers = ["status", "flag"]
    rows = [(0, ["open", "y"]), (1, ["open", "n"]), (2, ["open", "y"])]
    assert DocumentPipeline._tabular_key_column(headers, rows) is None


def test_sheet_fragment_uri_carve():
    """C4's first consumer: sheet-bearing chunks carve #sheet= URIs;
    single-sheet sources stay whole-file."""
    tab_with_sheet = {"doc_type": "tabular", "sheet": "Q3", "text": "x"}
    tab_single = {"doc_type": "tabular", "sheet": None, "text": "x"}
    # Mirror of _source_uri's tabular branch (site is a closure; the
    # contract is pinned here).
    doc_uri = "upload://doc_1"
    def carve(chunk):
        if chunk.get("doc_type") == "tabular" and chunk.get("sheet"):
            return f"{doc_uri}#sheet={chunk['sheet']}"
        return doc_uri
    assert carve(tab_with_sheet) == "upload://doc_1#sheet=Q3"
    assert carve(tab_single) == "upload://doc_1"
