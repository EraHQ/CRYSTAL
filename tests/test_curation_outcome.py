"""RC-18 / S12 (2026-10-08): a curation outcome is a closed set.

crystal_learn treated anything that was not exactly "fail" as success, so
"failed", "error", a typo or an empty string cached the pair as a good
answer. Now only "success" and "fail" are accepted; anything else is a
refusal with code bad_arguments and nothing is learned.

The fake service exposes exactly what crystal_learn calls on the real
LearningService: learn_from_failure(...) returning an object with
crystals_written / reflection / knowledge / category / error, and
cache_success(...) returning a bool.
"""
from types import SimpleNamespace

import pytest

from crystal_cache.agent.tools import curation


class _Svc:
    seen: list = []

    def __init__(self, **kw):
        pass

    async def learn_from_failure(self, **kw):
        _Svc.seen.append("fail")
        return SimpleNamespace(crystals_written=1, reflection="r", knowledge="k",
                               category="c", error=None)

    async def cache_success(self, **kw):
        _Svc.seen.append("success")
        return True


@pytest.fixture
def learn(monkeypatch):
    import crystal_cache.learning as learning

    _Svc.seen = []
    monkeypatch.setattr(learning, "LearningService", _Svc)
    monkeypatch.setattr(curation, "_get_state", lambda: {"store": None, "encoder": None, "vector_store": None})
    return getattr(curation.crystal_learn, "fn", curation.crystal_learn)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["failed", "error", "", "ok", "FAILURE", None])
async def test_unknown_outcome_is_refused_and_learns_nothing(bad, learn):
    out = await learn(customer_id="cus_x", prompt="p", response="r", outcome=bad)
    assert out.get("code") == "bad_arguments"
    assert _Svc.seen == []


@pytest.mark.asyncio
async def test_case_and_whitespace_are_tolerated_for_the_two_real_values(learn):
    await learn(customer_id="cus_x", prompt="p", response="r", outcome=" Fail ")
    await learn(customer_id="cus_x", prompt="p", response="r", outcome="SUCCESS")
    assert _Svc.seen == ["fail", "success"]
