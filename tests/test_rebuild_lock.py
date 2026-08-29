from __future__ import annotations

import pytest

from tools import rebuild_lock


def test_rebuild_lock_rejects_noncanonical_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rebuild_lock.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="must be rebuilt on Windows"):
        rebuild_lock.main()
