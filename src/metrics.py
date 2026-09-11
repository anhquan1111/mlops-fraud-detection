"""Đo lường các chỉ số vận hành hệ thống cho Prometheus (Request count, Latency, Fraud count)."""

import logging
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)
REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "fraud_http_requests_total",
    "Completed HTTP requests",
    ["method", "route", "status_class"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "fraud_http_request_duration_seconds",
    "HTTP response duration in seconds",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    registry=REGISTRY,
)
IN_FLIGHT = Gauge("fraud_http_requests_in_flight", "Active HTTP requests", registry=REGISTRY)


class HTTPMetricsMiddleware:
    """Count errors too, and avoid unbounded raw paths or transaction IDs as labels."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            method = "OTHER"
        started = time.perf_counter()
        status_code = 500  # An exception before response headers is still a failed request.
        IN_FLIGHT.inc()

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            duration = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", "unmatched")
            REQUESTS.labels(method, route, f"{status_code // 100}xx").inc()
            LATENCY.labels(method, route).observe(duration)
            IN_FLIGHT.dec()
            logger.info(
                "http_request method=%s route=%s status=%s duration_ms=%.3f",
                method,
                route,
                status_code,
                duration * 1000,
            )
