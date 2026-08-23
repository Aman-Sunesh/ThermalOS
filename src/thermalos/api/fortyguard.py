from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from thermalos.cache import JsonCache


class FortyGuardError(RuntimeError):
    pass


@dataclass
class FortyGuardResult:
    activity_id: str
    result: dict[str, Any]
    from_cache: bool = False
    elapsed_s: float | None = None


class FortyGuardClient:
    """FortyGuard client with caching and resilient polling.

    Design choices:
    - successful responses are cached by endpoint payload;
    - submission requests retry transient HTTP failures;
    - status 404 / 429 / 5xx and network errors retry the *same activity*;
    - Completed-with-empty results receive a short grace period;
    - only explicit terminal task failures may trigger a fresh submission;
    - processing timeouts never auto-resubmit because the original activity may
      still complete later;
    - optional verbose status logging makes long queues observable.
    """

    TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str = "data/cache/fortyguard",
        request_timeout_s: float = 90.0,
        poll_interval_s: float = 5.0,
        task_timeout_s: float = 900.0,
        verbose: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("FORTYGUARD_API_KEY")
        if not self.api_key:
            raise FortyGuardError("Set FORTYGUARD_API_KEY or pass api_key=...")
        self.base_url = (
            base_url or os.getenv("FORTYGUARD_BASE_URL") or "https://api.fortyguard.com"
        ).rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s
        self.task_timeout_s = task_timeout_s
        self.verbose = verbose
        self.cache = JsonCache(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {"api-key": self.api_key, "Content-Type": "application/json"}
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[FortyGuard] {message}", flush=True)

    def _request(
        self,
        method: str,
        path: str,
        *,
        retries: int = 3,
        **kwargs: Any,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        delay = 1.0
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = self.session.request(
                    method,
                    url,
                    timeout=self.request_timeout_s,
                    **kwargs,
                )
                if r.ok:
                    return r
                if r.status_code < 500 and r.status_code != 429:
                    raise FortyGuardError(
                        f"{method} {path} -> {r.status_code}: {r.text[:1000]}"
                    )
                last = FortyGuardError(
                    f"{method} {path} -> {r.status_code}: {r.text[:1000]}"
                )
            except requests.RequestException as exc:
                last = exc
            if attempt < retries:
                self._log(f"{method} {path} transient failure; retry {attempt}/{retries}")
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
        raise FortyGuardError(str(last) if last else f"{method} {path} failed")

    def _submit_and_wait(
        self,
        namespace: str,
        endpoint: str,
        payload: dict[str, Any],
        *,
        refresh: bool = False,
        required: Callable[[dict], bool] | None = None,
        terminal_retries: int = 2,
        completed_empty_grace_s: float = 45.0,
    ) -> FortyGuardResult:
        key = self.cache.key(namespace, payload)
        if not refresh:
            cached = self.cache.get(key)
            if cached is not None:
                self._log(f"cache hit: {namespace}")
                return FortyGuardResult(
                    activity_id=cached.get("activity_id", "cached"),
                    result=cached["result"],
                    from_cache=True,
                    elapsed_s=0.0,
                )

        last_failure: str | None = None

        for submission_attempt in range(1, terminal_retries + 1):
            started = time.monotonic()
            self._log(
                f"POST {endpoint} submission {submission_attempt}/{terminal_retries}"
            )
            body = self._request("POST", endpoint, json=payload).json()
            try:
                activity_id = body["data"]["activity_id"]
            except Exception as exc:
                raise FortyGuardError(f"Unexpected submission response: {body}") from exc

            self._log(f"activity_id={activity_id}")
            deadline = time.monotonic() + self.task_timeout_s
            poll_number = 0
            completed_empty_since: float | None = None

            while time.monotonic() < deadline:
                poll_number += 1
                try:
                    r = self.session.get(
                        f"{self.base_url}/v1/status/{activity_id}",
                        timeout=self.request_timeout_s,
                    )
                except requests.RequestException as exc:
                    self._log(
                        f"poll {poll_number:03d}: network error ({exc}); same activity"
                    )
                    time.sleep(self.poll_interval_s)
                    continue

                if r.status_code == 404:
                    self._log(
                        f"poll {poll_number:03d}: HTTP 404 not visible yet; same activity"
                    )
                    time.sleep(self.poll_interval_s)
                    continue

                if r.status_code in self.TRANSIENT_STATUS_CODES:
                    self._log(
                        f"poll {poll_number:03d}: HTTP {r.status_code} transient; same activity"
                    )
                    time.sleep(self.poll_interval_s)
                    continue

                if not r.ok:
                    raise FortyGuardError(
                        f"GET /v1/status/{activity_id} -> {r.status_code}: {r.text[:1000]}"
                    )

                try:
                    body = r.json()
                except ValueError:
                    self._log(
                        f"poll {poll_number:03d}: non-JSON status response; same activity"
                    )
                    time.sleep(self.poll_interval_s)
                    continue

                data = body.get("data", body)
                status = str(data.get("status", "")).lower()
                result = data.get("result") or {}
                tiles = len(result.get("map_data", {}).get("features", []))
                suffix = f" tiles={tiles}" if tiles else ""
                self._log(f"poll {poll_number:03d}: {status or 'unknown'}{suffix}")

                if status in {"failed", "error", "cancelled"}:
                    last_failure = f"Activity {activity_id} ended as {status}: {data}"
                    break

                if status in {"completed", "succeeded"}:
                    if result and (required is None or required(result)):
                        elapsed = time.monotonic() - started
                        self.cache.put(
                            key,
                            {"activity_id": activity_id, "result": result},
                        )
                        self._log(f"ready in {elapsed:.1f}s")
                        return FortyGuardResult(
                            activity_id,
                            result,
                            False,
                            elapsed,
                        )

                    if completed_empty_since is None:
                        completed_empty_since = time.monotonic()
                        self._log("Completed but result is empty; grace period started")
                    elif (
                        time.monotonic() - completed_empty_since
                        >= completed_empty_grace_s
                    ):
                        last_failure = (
                            f"Activity {activity_id} remained completed with no usable "
                            f"result for {completed_empty_grace_s:.0f}s"
                        )
                        break
                else:
                    completed_empty_since = None

                time.sleep(self.poll_interval_s)

            else:
                raise FortyGuardError(
                    f"Activity {activity_id} was still unfinished after "
                    f"{self.task_timeout_s:.0f}s. Not resubmitting automatically."
                )

            if submission_attempt < terminal_retries:
                wait_s = min(2 ** (submission_attempt - 1), 4)
                self._log(f"terminal failure; fresh resubmission in {wait_s}s")
                time.sleep(wait_s)

        raise FortyGuardError(last_failure or "FortyGuard task failed after retries")

    @staticmethod
    def _heatmap_ready(result: dict) -> bool:
        return bool(result.get("map_data", {}).get("features"))

    def heatmap(
        self,
        *,
        polygon_aoi: dict,
        start_date: str,
        filter_type: int,
        granularity: int = 100,
        start_time: str | None = None,
        end_time: str | None = None,
        end_date: str | None = None,
        analytic_type: str = "tcm",
        threshold: float | None = None,
        direction: str = "above",
        refresh: bool = False,
        omit_explicit_tcm: bool = True,
    ) -> FortyGuardResult:
        date_time: dict[str, Any] = {
            "start_date": start_date,
            "filter_type": filter_type,
        }
        if start_time is not None:
            date_time["start_time"] = start_time
        if end_time is not None:
            date_time["end_time"] = end_time
        if end_date is not None:
            date_time["end_date"] = end_date
        payload: dict[str, Any] = {
            "polygon_aoi": polygon_aoi,
            "date_time": date_time,
            "granularity": int(granularity),
        }
        if not (analytic_type == "tcm" and omit_explicit_tcm):
            payload["analytic_type"] = analytic_type
        if analytic_type in {"exceedance", "persistence"}:
            if threshold is None:
                raise ValueError(f"{analytic_type} requires threshold in °C")
            payload["threshold"] = float(threshold)
            payload["direction"] = direction
        return self._submit_and_wait(
            f"heatmap_{analytic_type}",
            "/v1/heatmap",
            payload,
            refresh=refresh,
            required=self._heatmap_ready,
        )

    def environmental_parameters(
        self,
        *,
        latitude: float,
        longitude: float,
        temperature_c: float,
        start_date: str,
        start_time: str,
        filter_type: int = 1,
        refresh: bool = False,
    ) -> FortyGuardResult:
        payload = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "temperature": float(temperature_c),
            "date_time": {
                "start_date": start_date,
                "start_time": start_time,
                "filter_type": int(filter_type),
            },
        }
        return self._submit_and_wait(
            "env",
            "/v1/env_params",
            payload,
            refresh=refresh,
            required=lambda r: bool(r.get("locations")),
        )

    def satellite(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        start_time: str,
        granularity: int = 100,
        refresh: bool = False,
    ) -> FortyGuardResult:
        payload = {
            "sat": {"latitude": float(latitude), "longitude": float(longitude)},
            "date_time": {
                "start_date": start_date,
                "start_time": start_time,
                "filter_type": 1,
            },
            "granularity": int(granularity),
        }
        return self._submit_and_wait(
            "satellite",
            "/v1/satellite",
            payload,
            refresh=refresh,
            required=lambda r: bool(r.get("segmentation")),
        )

    def streetview(
        self,
        *,
        latitude: float,
        longitude: float,
        horizontal_angle: float = 0.0,
        vertical_angle: float = 0.0,
        back_view: bool = False,
        refresh: bool = False,
    ) -> FortyGuardResult:
        payload = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "vertical_angle": float(vertical_angle),
            "horizontal_angle": float(horizontal_angle),
            "back_view": bool(back_view),
        }
        return self._submit_and_wait(
            "streetview",
            "/v1/streetview",
            payload,
            refresh=refresh,
            required=lambda r: bool(r.get("front")),
        )

    def usage(self) -> dict[str, Any]:
        paths = [
            "/v1/system/fetch-api-key-usage",
            "/v1/system/fetch-api-key-custom-usage",
        ]
        errors = []
        for path in paths:
            try:
                payload = {} if path.endswith("usage") else None
                r = (
                    self._request("POST", path, json=payload)
                    if payload is not None
                    else self._request("POST", path)
                )
                return r.json()
            except Exception as exc:
                errors.append(str(exc))
        raise FortyGuardError("Usage lookup failed: " + " | ".join(errors))
