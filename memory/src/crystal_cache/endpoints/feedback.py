"""Feedback endpoint — POST /v1/feedback.

Records a thumbs-up/down on a specific assistant turn and triggers
the LearningService to learn from the feedback signal.

Phase 6 wrote the Feedback row only; Phase 7 Wave 7F adds the
learning trigger inline now that `learning.LearningService` has
ported (Wave 7E).

The trigger fires when body.signal is "up" or "down":
  - "down": LearningService.learn_from_failure — extracts a
    reflection rule + knowledge crystal via Level B+F from the
    QueryLog's prompt + response + user's failure signal.
  - "up": LearningService.cache_success — writes the assistant's
    answer as a cached_solution crystal so future identical
    queries hit the cache.

Both fire-and-forget: a learning failure logs the error but does
NOT fail the feedback response (the row was already written). The
FeedbackResponse carries learning_triggered + crystals_written so
SDK consumers can observe whether learning ran.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..ingress.schema import FeedbackRequest, FeedbackResponse
from ..models import Customer, Feedback

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> FeedbackResponse:
    """Record a thumbs-up/down on a specific assistant turn.

    Customer identity comes from the Bearer token, not the body —
    this prevents one customer from submitting feedback against
    another customer's conversations.

    After the row is written, the learning trigger fires:
      - thumbs-down → learn_from_failure (reflection + knowledge)
      - thumbs-up   → cache_success (cached_solution crystal)

    Learning is fire-and-forget; failures log but don't fail the
    feedback response. Per v1's pattern: the feedback was already
    recorded successfully; learning is a separate concern.
    """
    feedback = Feedback(
        id=f"fb_{uuid.uuid4().hex[:16]}",
        customer_id=customer.id,
        sequence_id=body.sequence_id,
        turn_index=body.turn_index,
        signal=body.signal,
        comment=body.comment,
        created_at=datetime.now(timezone.utc),
    )
    try:
        await store.write_feedback(feedback)
    except Exception as e:
        logger.error(
            "feedback.write_failed",
            customer_id=customer.id,
            sequence_id=body.sequence_id,
            turn_index=body.turn_index,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="failed to record feedback",
        )

    logger.info(
        "feedback.recorded",
        customer_id=customer.id,
        feedback_id=feedback.id,
        sequence_id=body.sequence_id,
        turn_index=body.turn_index,
        signal=body.signal,
        has_comment=body.comment is not None,
    )

    # ---- Wave 7F: Trigger learning from feedback ----
    #
    # Look up the QueryLog for this turn, extract prompt + response,
    # and dispatch to LearningService. Fire-and-forget — learning
    # failures don't fail the feedback response.
    learning_triggered = False
    crystals_written = 0

    if body.signal in ("up", "down"):
        try:
            query_log = await store.find_query_log_by_sequence(
                customer_id=customer.id,
                sequence_id=body.sequence_id,
                turn_index=body.turn_index,
            )
            if query_log is not None:
                from ..learning import LearningService
                encoder = request.app.state.prompt_encoder
                vector_store = request.app.state.vector_store
                learning_svc = LearningService(
                    store=store,
                    encoder=encoder,
                    vector_store=vector_store,
                    vector_index=getattr(request.app.state, "vector_index", None),
                )

                if body.signal == "down":
                    failure_signal = (
                        body.comment
                        or "User indicated this response was incorrect"
                    )
                    result = await learning_svc.learn_from_failure(
                        customer_id=customer.id,
                        prompt=query_log.query_text,
                        response=query_log.response_text or "",
                        failure_signal=failure_signal,
                        crystal_type="customer:legacy",
                    )
                    learning_triggered = True
                    crystals_written = result.crystals_written
                    logger.info(
                        "feedback.learning_triggered",
                        customer_id=customer.id,
                        signal="down",
                        crystals_written=result.crystals_written,
                        reflection=(
                            result.reflection[:80]
                            if result.reflection
                            else None
                        ),
                    )
                elif body.signal == "up":
                    cached = await learning_svc.cache_success(
                        customer_id=customer.id,
                        prompt=query_log.query_text,
                        solution=query_log.response_text or "",
                        crystal_type="customer:legacy",
                    )
                    learning_triggered = cached
                    crystals_written = 1 if cached else 0
                    logger.info(
                        "feedback.cache_success",
                        customer_id=customer.id,
                        signal="up",
                        cached=cached,
                    )
        except Exception as e:
            # Learning failure should NOT fail the feedback response.
            # The feedback was already recorded; learning is best-effort.
            logger.error(
                "feedback.learning_failed",
                customer_id=customer.id,
                error=str(e),
                error_type=type(e).__name__,
            )

    return FeedbackResponse(
        id=feedback.id,
        customer_id=feedback.customer_id,
        sequence_id=feedback.sequence_id,
        turn_index=feedback.turn_index,
        signal=feedback.signal,
        comment=feedback.comment,
        created_at=feedback.created_at.isoformat(),
        learning_triggered=learning_triggered,
        crystals_written=crystals_written,
    )
