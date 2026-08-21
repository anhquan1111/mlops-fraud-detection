"""Unit tests for src/config.py — the DECISION_THRESHOLD environment override.

The threshold is the system's operating point. It is read from the environment
so the operations team can move it without a code change, an image rebuild, or a
retrain — so the override has to actually work, and a malformed value has to
fail loudly rather than quietly serving the default.

Each test reloads src.config so it observes the environment as it would at
process startup.
"""

from __future__ import annotations

import importlib

import pytest

import src.config


def _reload_with(monkeypatch: pytest.MonkeyPatch, value: str | None):
    """Reload src.config with DECISION_THRESHOLD set to `value` (None = unset).

    Args:
        monkeypatch: pytest fixture used to scope the environment change.
        value: Raw environment value, or None to delete the variable.

    Returns:
        The freshly reloaded src.config module.
    """
    if value is None:
        monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("DECISION_THRESHOLD", value)
    return importlib.reload(src.config)


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload src.config after each test so no override leaks into other tests."""
    yield
    importlib.reload(src.config)


class TestDecisionThresholdOverride:
    def test_env_var_changes_the_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: setting the variable moves the operating point."""
        config = _reload_with(monkeypatch, "0.81")
        assert config.DECISION_THRESHOLD == pytest.approx(0.81)

    def test_default_is_neutral_half(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset means 0.50 — the deliberately untuned default."""
        config = _reload_with(monkeypatch, None)
        assert config.DECISION_THRESHOLD == pytest.approx(0.5)

    def test_blank_value_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty or whitespace value is treated as unset, not as an error."""
        config = _reload_with(monkeypatch, "   ")
        assert config.DECISION_THRESHOLD == pytest.approx(0.5)

    @pytest.mark.parametrize("raw", ["abc", "0.81x", " 0.5.1 ", "81%"])
    def test_malformed_value_raises(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """A non-numeric value must fail loudly, never silently serve 0.5.

        An operator who sets a malformed threshold intended to change the
        operating point; quietly ignoring them would hide that.
        """
        with pytest.raises(ValueError, match="DECISION_THRESHOLD"):
            _reload_with(monkeypatch, raw)

    @pytest.mark.parametrize("raw", ["0", "0.0", "1", "1.0", "1.5", "-0.2"])
    def test_out_of_range_value_raises(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """0 would flag every transaction and 1 would flag none — both rejected."""
        with pytest.raises(ValueError, match="out of range"):
            _reload_with(monkeypatch, raw)

    def test_boundary_values_are_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values just inside the open interval (0, 1) are valid."""
        for raw in ("0.01", "0.99"):
            config = _reload_with(monkeypatch, raw)
            assert config.DECISION_THRESHOLD == pytest.approx(float(raw))

    def test_evaluate_uses_the_overridden_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Downstream code must pick the override up, not a stale 0.5.

        evaluate_model() defaults its `threshold` argument from config, so a
        module reloaded with an override has to flow through to scoring.
        """
        import numpy as np
        import pandas as pd

        config = _reload_with(monkeypatch, "0.90")

        class _StubModel:
            """Returns a fixed fraud probability of 0.80 for every row."""

            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.2), np.full(len(X), 0.8)])

        import src.evaluate

        evaluate = importlib.reload(src.evaluate)

        X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([1, 1, 0, 0])

        # At 0.90 nothing clears the bar, so no positive is predicted at all.
        metrics = evaluate.evaluate_model(_StubModel(), X, y)
        assert metrics["threshold"] == pytest.approx(0.90)
        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert config.DECISION_THRESHOLD == pytest.approx(0.90)

        # Restore src.evaluate against the default config for later tests.
        _reload_with(monkeypatch, None)
        importlib.reload(src.evaluate)
