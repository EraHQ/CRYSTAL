"""BaaTracking — HIPAA Business Associate Agreement state per customer.

One row per customer (unique constraint on customer_id). Tracks
whether a BAA has been signed and what data sources the customer has
told us contain PHI. The flag itself doesn't grant access — it
records compliance posture for audit and surfaces in the inspector's
compliance view.

phi_data_sources is a free-form list of identifiers the customer
provided (e.g. ['google_drive:medical_records_folder',
'documents:patient_intake']). The system doesn't parse or validate
these; they're descriptive labels for the auditor's benefit. The
authoritative PHI flag lives on WatchedFolder.contains_phi and
WatchedFile.contains_phi.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class BaaTracking(BaseModel):
    id: str
    customer_id: str  # unique — one row per customer

    baa_signed: bool = False
    baa_signed_date: Optional[datetime] = None
    baa_document_ref: Optional[str] = None  # URL or doc-store reference

    phi_data_sources: Optional[list[str]] = None
    hipaa_contact_email: Optional[str] = None
    notes: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
