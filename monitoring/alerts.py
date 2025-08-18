"""Alert evaluation engine for TradeBot monitoring.

Loads alert rules from ``alerts.yml`` and evaluates them either against
local Prometheus metrics or by issuing PromQL queries to a remote
Prometheus instance. When an alert condition is met, configurable hooks
are executed (logging, email, webhooks, ...).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import httpx
import yaml
from prometheus_client import REGISTRY

logger = logging.getLogger(__name__)

# Default path to the alerts configuration file residing next to this module
DEFAULT_CONFIG = Path(__file__).with_name("alerts.yml")


class AlertManager:
    """Evaluate alert rules defined in ``alerts.yml``.

    Parameters
    ----------
    config_path:
        Path to the YAML file containing alert rule definitions.  By
        default ``monitoring/alerts.yml`` is used.
    prometheus_url:
        Base URL for a Prometheus server used to evaluate PromQL
        expressions that cannot be resolved locally.
    hooks:
        Optional mapping defining hooks to be executed when an alert
        triggers.  Supported keys are ``log`` (bool), ``email`` (callable)
        and ``webhook`` (URL string or callable).
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        prometheus_url: str | None = None,
        hooks: Dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        self.prometheus_url = prometheus_url or "http://localhost:9090"
        self.hooks = hooks or {"log": True}
        self.rules = self._load_rules()

    # ------------------------------------------------------------------
    def _load_rules(self) -> list[dict]:
        if not self.config_path.exists():
            logger.warning("Alert config %s not found", self.config_path)
            return []
        with self.config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        groups = data.get("groups", [])
        return [rule for g in groups for rule in g.get("rules", [])]

    # ------------------------------------------------------------------
    def _metric_context(self) -> Dict[str, float]:
        """Return a mapping of metric names to their current values."""

        context: Dict[str, float] = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                # Only expose samples without labels to keep evaluation simple
                if sample.labels:
                    continue
                context[sample.name] = sample.value
        return context

    # ------------------------------------------------------------------
    def _eval_local(self, expr: str) -> bool:
        """Evaluate a simple expression using current metric values."""

        context = self._metric_context()
        try:
            return bool(eval(expr, {"__builtins__": {}}, context))
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug("Local alert evaluation failed for %s: %s", expr, exc)
            return False

    # ------------------------------------------------------------------
    def _eval_promql(self, expr: str) -> bool:
        """Query Prometheus and interpret the result as a boolean."""

        try:
            resp = httpx.get(
                f"{self.prometheus_url}/api/v1/query", params={"query": expr}, timeout=5.0
            )
            resp.raise_for_status()
            results = resp.json().get("data", {}).get("result", [])
            for item in results:
                try:
                    value = float(item.get("value", [0, "0"])[1])
                    if value:
                        return True
                except (TypeError, ValueError):
                    continue
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.debug("PromQL evaluation failed for %s: %s", expr, exc)
        return False

    # ------------------------------------------------------------------
    def _emit_hooks(self, rule: dict) -> None:
        if self.hooks.get("log"):
            logger.warning("Alert triggered: %s", rule.get("alert"))
        email_hook = self.hooks.get("email")
        if callable(email_hook):  # pragma: no cover - optional
            try:
                email_hook(rule)
            except Exception:  # pragma: no cover - best effort
                logger.exception("Email hook failed for %s", rule.get("alert"))
        webhook = self.hooks.get("webhook")
        if webhook:
            try:
                if callable(webhook):
                    webhook(rule)
                else:
                    httpx.post(webhook, json=rule, timeout=5.0)
            except Exception:  # pragma: no cover - best effort
                logger.exception("Webhook hook failed for %s", rule.get("alert"))

    # ------------------------------------------------------------------
    def check(self) -> list[dict]:
        """Evaluate all rules and return a list of active alerts."""

        active: list[dict] = []
        for rule in self.rules:
            expr = rule.get("expr", "")
            # First try evaluating locally; if it returns False and the
            # expression looks non-trivial, fall back to PromQL.
            triggered = self._eval_local(expr)
            if not triggered and any(ch in expr for ch in ["[", "("]):
                triggered = self._eval_promql(expr)
            if triggered:
                self._emit_hooks(rule)
                active.append(
                    {
                        "labels": {
                            "alertname": rule.get("alert"),
                            **rule.get("labels", {}),
                        },
                        "annotations": rule.get("annotations", {}),
                        "expr": expr,
                    }
                )
        return active


# Convenience function used by the monitoring panel
ALERT_MANAGER = AlertManager()


def evaluate_alerts() -> list[dict]:
    """Return currently active alerts using the default manager."""

    return ALERT_MANAGER.check()
