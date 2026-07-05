"""Document — §6 of BUILD_PROPOSAL.md.

Source material ingested by the cache. Facts are extracted from documents
and verified before being written to crystals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


DocumentSource = Literal["upload", "rag_retrieval", "api"]


class Document(BaseModel):
    id: str
    customer_id: str

    source: DocumentSource
    content: str

    facts_extracted_count: int = 0
    facts_verified_count: int = 0
    facts_rejected_count: int = 0

    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
