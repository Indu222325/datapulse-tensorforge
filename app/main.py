"""Data Pulse - RideEat support ticket classification service (TensorForge 2.0, Phase 2).

Implements specs/tensorforge-phase2-openapi-v2.yaml:

    GET    /health                      public readiness probe
    POST   /predict                     one ticket
    POST   /predict/batch               1-100 tickets, synchronous, atomic
    POST   /batch/jobs                  1-5,000 tickets, asynchronous
    GET    /batch/jobs/{job_id}         poll status
    GET    /batch/jobs/{job_id}/results fetch (paged) results
    DELETE /batch/jobs/{job_id}         cancel / discard
    GET    /                            demo site + dashboard (static page, calls the API above)

Order of checks on every protected endpoint: auth (401) -> content type (415) -> size (413)
-> JSON parse (400) -> validation (422). Request bodies are read and validated by hand rather
than through FastAPI/pydantic models so that this order and the error format are exact.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse, Response

from .jobs import JobStore, JobWorker, QueueFull
from .model import TicketModel
from .validation import clean_ticket, validate_batch, validate_ticket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("datapulse")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

LIMIT_PREDICT = 1 * 1024 * 1024       # ~1 MB
LIMIT_BATCH = 5 * 1024 * 1024         # 5 MB
LIMIT_JOB = 25 * 1024 * 1024          # 25 MB
MAX_BATCH = 100
MAX_JOB = 5000
POLL_SECONDS = "2"
DRAIN_CAP = 64 * 1024 * 1024       # stop reading absurdly large uploads after this many bytes


def _load_dotenv() -> None:
    """Local convenience only: read KEY=VALUE lines from .env if present (never copied into the image)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class State:
    api_key: str = ""
    model: TicketModel | None = None
    store: JobStore | None = None
    worker: JobWorker | None = None


state = State()


# ----------------------------------------------------------------------------- responses / errors
class JSON(Response):
    """ASCII-only JSON: safe even if a client sends lone surrogates or other odd Unicode in ticket_id."""
    media_type = "application/json"

    def __init__(self, content, status_code: int = 200, headers: dict | None = None):
        body = json.dumps(content, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        super().__init__(body, status_code=status_code, headers=headers, media_type="application/json")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: list | None = None, headers: dict | None = None):
        self.status, self.code, self.message, self.details, self.headers = status, code, message, details, headers


def error_body(code: str, message: str, details: list | None = None) -> dict:
    err = {"code": code, "message": message}
    if details:
        err["details"] = details
    return {"error": err}


# ----------------------------------------------------------------------------- auth + body parsing
def require_auth(request: Request) -> None:
    """X-API-Key or Authorization: Bearer. Constant-time compare. No key configured -> always 401."""
    presented = []
    key = request.headers.get("x-api-key")
    if key is not None:
        presented.append(key.strip())
    auth = request.headers.get("authorization")
    if auth is not None and auth[:7].lower() == "bearer ":
        presented.append(auth[7:].strip())
    expected = state.api_key.encode("utf-8")
    ok = bool(expected) and any(hmac.compare_digest(p.encode("utf-8", "ignore"), expected) for p in presented if p)
    if not ok:
        message = "Invalid API key." if any(presented) else "Missing API key. Send X-API-Key or Authorization Bearer."
        raise ApiError(401, "unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


async def read_json(request: Request, limit: int):
    """Auth -> content type -> size -> JSON parse, in the contract's order.

    The body is always read off the socket (up to DRAIN_CAP) before an error is returned. Answering
    early and closing the connection makes many HTTP clients report "connection reset" instead of
    the 401 / 413 / 415 we actually sent.
    """
    error: ApiError | None = None
    try:
        require_auth(request)
    except ApiError as exc:
        error = exc
    if error is None and request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        error = ApiError(415, "unsupported_media_type", "Content-Type must be application/json.")
    too_large = ApiError(413, "payload_too_large", "Request body exceeds the maximum allowed size.")
    declared = request.headers.get("content-length")
    if error is None and declared and declared.isdigit() and int(declared) > limit:
        error = too_large

    body, received = bytearray(), 0
    async for chunk in request.stream():
        received += len(chunk)
        if error is None:
            if received > limit:
                error, body = too_large, bytearray()
            else:
                body.extend(chunk)
        if received > DRAIN_CAP:
            break
    if error is not None:
        raise error
    try:
        return json.loads(bytes(body))
    except (ValueError, RecursionError):
        raise ApiError(400, "malformed_json", "Request body is not valid JSON.") from None


def get_model() -> TicketModel:
    if state.model is None:
        raise ApiError(503, "model_loading", "Model is not loaded yet. Retry shortly.", headers={"Retry-After": "5"})
    return state.model


def validation_error(details: list) -> ApiError:
    return ApiError(422, "validation_error", "Request failed validation.", details)


# ----------------------------------------------------------------------------- app + lifecycle
def _startup_load() -> None:
    try:
        model = TicketModel()
        model.predict([{"channel": "chat", "subject": "", "text": "warm up"}])
        state.worker = JobWorker(state.store, model.predict)
        state.worker.start()
        state.model = model
        log.info("model %s loaded", model.version)
    except Exception:
        log.exception("model failed to load")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_dotenv()
    state.api_key = os.environ.get("API_KEY", "").strip()
    if not state.api_key:
        log.warning("API_KEY is not set: every protected endpoint will answer 401.")
    data_dir = Path(os.environ.get("DATA_DIR", ROOT / "runtime"))
    retention = max(int(os.environ.get("JOB_RETENTION_SECONDS", 24 * 3600)), 6 * 3600)
    state.store = JobStore(data_dir / "jobs.sqlite3", retention)
    interrupted = state.store.recover_after_restart()
    if interrupted:
        log.warning("%d job(s) were running during the last shutdown and are now marked failed/interrupted", interrupted)
    state.model = None
    threading.Thread(target=_startup_load, name="model-loader", daemon=True).start()
    yield
    if state.worker is not None:
        state.worker.stop()


app = FastAPI(title="Data Pulse - RideEat ticket classifier", version="1.0.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)


class RequestIdMiddleware:
    """Echo X-Request-ID on every response, including errors."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        request_id = next((v[:128] for k, v in scope.get("headers", []) if k == b"x-request-id"), None)

        async def send_with_id(message):
            if request_id and message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id))
                message = {**message, "headers": headers}
            await send(message)

        await self.inner(scope, receive, send_with_id)


app.add_middleware(RequestIdMiddleware)


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return JSON(error_body(exc.code, exc.message, exc.details), exc.status, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    codes = {404: ("not_found", "Unknown route."), 405: ("method_not_allowed", "Method not allowed on this route.")}
    code, message = codes.get(exc.status_code, ("http_error", "Request could not be processed."))
    return JSON(error_body(code, message), exc.status_code, getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _request_validation(_: Request, __: RequestValidationError):
    return JSON(error_body("validation_error", "Request failed validation."), 422)


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("unhandled error: %s", type(exc).__name__)
    return JSON(error_body("internal_error", "Unexpected server error."), 500)


# ----------------------------------------------------------------------------- core endpoints
@app.get("/health")
async def health():
    if state.model is None:
        return JSON({"status": "loading", "model_version": None, "model_loaded": False}, 503, {"Retry-After": "5"})
    return JSON({"status": "ok", "model_version": state.model.version, "model_loaded": True})


@app.post("/predict")
async def predict(request: Request):
    body = await read_json(request, LIMIT_PREDICT)
    problems = validate_ticket(body)
    if problems:
        raise validation_error(problems)
    model = get_model()
    result = await run_in_threadpool(model.predict, [clean_ticket(body)])
    return JSON(result[0])


@app.post("/predict/batch")
async def predict_batch(request: Request):
    body = await read_json(request, LIMIT_BATCH)
    problems = validate_batch(body, MAX_BATCH)
    if problems:
        raise validation_error(problems)
    model = get_model()
    started = time.perf_counter()
    predictions = await run_in_threadpool(model.predict, [clean_ticket(t) for t in body["tickets"]])
    return JSON({
        "predictions": predictions,
        "meta": {"count": len(predictions), "model_version": model.version,
                 "processing_time_ms": int((time.perf_counter() - started) * 1000)},
    })


# ----------------------------------------------------------------------------- async batch jobs
@app.post("/batch/jobs")
async def submit_job(request: Request):
    body = await read_json(request, LIMIT_JOB)
    problems = validate_batch(body, MAX_JOB)
    idem_key = request.headers.get("idempotency-key")
    if idem_key is not None and len(idem_key) > 128:
        problems.append({"field": "Idempotency-Key", "issue": "must be at most 128 characters"})
    if problems:
        raise validation_error(problems)
    model = get_model()
    tickets = [clean_ticket(t) for t in body["tickets"]]
    try:
        status, _created = await run_in_threadpool(state.store.submit, tickets, model.version, idem_key or None)
    except QueueFull:
        raise ApiError(429, "too_many_jobs", "Job queue is full. Retry later.", headers={"Retry-After": "10"}) from None
    if state.worker is not None:
        state.worker.wake.set()
    return JSON(status, 202, {"Location": f"/batch/jobs/{status['job_id']}", "Retry-After": POLL_SECONDS})


def _find_job(job_id: str, expired_as_gone: bool = True) -> dict:
    kind, status = state.store.lookup(job_id) if len(job_id) <= 128 else ("missing", None)
    if kind == "expired" and expired_as_gone:
        raise ApiError(410, "job_expired", "Job results have expired.")
    if status is None:
        raise ApiError(404, "job_not_found", "No job with this id.")
    return status


@app.get("/batch/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    require_auth(request)
    status = await run_in_threadpool(_find_job, job_id)
    headers = {"Retry-After": POLL_SECONDS} if status["status"] in ("queued", "running") else None
    return JSON(status, 200, headers)


@app.delete("/batch/jobs/{job_id}")
async def job_delete(job_id: str, request: Request):
    require_auth(request)
    await run_in_threadpool(_find_job, job_id, False)
    await run_in_threadpool(state.store.delete, job_id)
    return Response(status_code=204)


def _int_param(request: Request, name: str, minimum: int, maximum: int | None):
    raw = request.query_params.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or value < minimum or (maximum is not None and value > maximum):
        bound = f"an integer >= {minimum}" if maximum is None else f"an integer between {minimum} and {maximum}"
        raise validation_error([{"field": name, "issue": f"must be {bound}"}])
    return value


@app.get("/batch/jobs/{job_id}/results")
async def job_results(job_id: str, request: Request):
    require_auth(request)
    status = await run_in_threadpool(_find_job, job_id)
    if status["status"] != "succeeded":
        raise ApiError(409, "job_not_ready",
                       f"Job status is '{status['status']}'. Results are available once it is 'succeeded'.")
    offset = _int_param(request, "offset", 0, None) or 0
    limit = _int_param(request, "limit", 1, MAX_JOB)
    predictions = await run_in_threadpool(state.store.results, job_id)
    if predictions is None:
        raise ApiError(404, "job_not_found", "No job with this id.")
    total = len(predictions)
    if limit is None:
        limit = max(total - offset, 1)
    page = predictions[offset:offset + limit]
    end = offset + len(page)
    return JSON({
        "job_id": job_id, "status": "succeeded", "total": total, "offset": offset, "limit": limit,
        "next_offset": end if page and end < total else None,
        "model_version": status["model_version"], "predictions": page,
    })


# ----------------------------------------------------------------------------- demo site (not part of the scored contract)
@app.get("/")
@app.get("/demo")
async def demo_page():
    return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


@app.get("/demo/metrics")
async def demo_metrics():
    """Public: validation metrics and experiment history shown on the dashboard tab."""
    model = get_model()
    card_path = model.dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    return JSON({"metrics": model.metrics, "model_card": card})


@app.post("/demo/explain")
async def demo_explain(request: Request):
    """Protected: prediction plus top-3 categories and matched concepts for the demo UI."""
    body = await read_json(request, LIMIT_PREDICT)
    problems = validate_ticket(body)
    if problems:
        raise validation_error(problems)
    model = get_model()
    ticket = clean_ticket(body)
    prediction, explanation = await run_in_threadpool(lambda: (model.predict([ticket])[0], model.explain(ticket)))
    return JSON({"prediction": prediction, "explanation": explanation})
