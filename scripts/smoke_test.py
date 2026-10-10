"""End-to-end check of a RUNNING service (local Docker container or the hosted URL).

    python scripts/smoke_test.py http://localhost:8000 YOUR_API_KEY
    python scripts/smoke_test.py https://your-name-datapulse.hf.space YOUR_API_KEY

Uses only the Python standard library. If the `jsonschema` package is installed, every
response is also validated against the organisers' schemas in specs/schemas.
It ends with "ALL CHECKS PASSED" or a list of what failed.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) < 3:
    sys.exit("usage: python scripts/smoke_test.py <base_url> <api_key> [job_size]")
BASE, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
JOB_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2500

try:
    from jsonschema import Draft202012Validator
    SCHEMAS = {p.name.replace(".schema.json", ""): Draft202012Validator(json.loads(p.read_text(encoding="utf-8")))
               for p in (ROOT / "specs" / "schemas").glob("*.schema.json")}
except ImportError:
    SCHEMAS = {}
    print("note: jsonschema is not installed, so responses are not schema-validated (pip install jsonschema)\n")

failures: list[str] = []


def call(method: str, path: str, body=None, key: str | None = KEY, headers: dict | None = None, raw: bytes | None = None):
    h = dict(headers or {})
    if key is not None:
        h.setdefault("X-API-Key", key)
    data = raw
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status, text, rh = resp.status, resp.read().decode("utf-8"), resp.headers
    except urllib.error.HTTPError as err:
        status, text, rh = err.code, err.read().decode("utf-8"), err.headers
    try:
        payload = json.loads(text) if text else None
    except ValueError:
        payload = text
    return status, payload, rh


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   -> {detail}"))
    if not ok:
        failures.append(name)


def valid(schema: str, payload) -> tuple[bool, str]:
    if schema not in SCHEMAS:
        return True, ""
    errors = [e.message for e in SCHEMAS[schema].iter_errors(payload)]
    return not errors, "; ".join(errors[:2])


def expect(name: str, result, status: int, schema: str | None = None):
    code, payload, _ = result
    ok, why = code == status, f"expected {status}, got {code}: {str(payload)[:160]}"
    if ok and schema:
        ok, why = valid(schema, payload)
    if ok and status >= 400:
        ok, why = valid("error_response", payload)
    check(name, ok, why)
    return payload


def load(name: str) -> list[dict]:
    path = ROOT / "data" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_request(row: dict, ticket_id: str | None = None) -> dict:
    return {"ticket_id": ticket_id or row["ticket_id"], "channel": row["channel"], "subject": row["subject"], "text": row["text"]}


print(f"Target: {BASE}\n")

# --- health -----------------------------------------------------------------
health = expect("GET /health is public and ready", call("GET", "/health", key=None), 200, "health_response")
version = (health or {}).get("model_version")

# --- auth -------------------------------------------------------------------
one = {"ticket_id": "SMOKE-1", "channel": "chat", "subject": "", "text": "bro mage order eka hour ekakata wada late, refund ekak denna"}
expect("no key -> 401", call("POST", "/predict", one, key=None), 401)
expect("wrong key -> 401", call("POST", "/predict", one, key="wrong"), 401)
expect("no key + broken body is still 401", call("POST", "/predict", raw=b"{oops", key=None, headers={"Content-Type": "application/json"}), 401)
expect("Authorization: Bearer works", call("POST", "/predict", one, key=None, headers={"Authorization": f"Bearer {KEY}"}), 200, "predict_response")

# --- predict ----------------------------------------------------------------
code, pred, rh = call("POST", "/predict", one, headers={"X-Request-ID": "smoke-42"})
expect("POST /predict -> 200", (code, pred, rh), 200, "predict_response")
check("ticket_id echoed", (pred or {}).get("ticket_id") == "SMOKE-1")
check("X-Request-ID echoed", rh.get("X-Request-ID") == "smoke-42", str(rh.get("X-Request-ID")))
check("model_version same as /health", (pred or {}).get("model_version") == version)
expect("emoji-only text is valid", call("POST", "/predict", {"channel": "chat", "text": "😡😡"}), 200, "predict_response")
expect("empty text -> 422", call("POST", "/predict", {"channel": "chat", "text": "   "}), 422)
expect("bad channel -> 422", call("POST", "/predict", {"channel": "fax", "text": "hi"}), 422)
expect("missing field -> 422", call("POST", "/predict", {"text": "hi"}), 422)
expect("malformed JSON -> 400", call("POST", "/predict", raw=b"{oops", headers={"Content-Type": "application/json"}), 400)
expect("wrong content type -> 415", call("POST", "/predict", raw=json.dumps(one).encode(), headers={"Content-Type": "text/plain"}), 415)
expect("oversized body -> 413", call("POST", "/predict", {"channel": "chat", "text": "x" * 1_100_000}), 413)
expect("unknown route -> JSON 404", call("GET", "/definitely-not-here"), 404)
expect("wrong method -> JSON 405", call("GET", "/predict"), 405)

# --- sync batch ---------------------------------------------------------------
val = load("validation")
batch = [as_request(r) for r in val[:100]]
code, out, _ = call("POST", "/predict/batch", {"tickets": batch})
expect("POST /predict/batch (100 tickets) -> 200", (code, out, None), 200, "batch_response")
check("batch keeps order and ids", [p["ticket_id"] for p in (out or {}).get("predictions", [])] == [t["ticket_id"] for t in batch])
expect("101 tickets -> 422", call("POST", "/predict/batch", {"tickets": batch + [as_request(val[100])]}), 422)
expect("duplicate ticket_id -> 422", call("POST", "/predict/batch", {"tickets": [batch[0], batch[0]]}), 422)
expect("one bad item fails whole batch", call("POST", "/predict/batch", {"tickets": [batch[0], {"ticket_id": "x", "channel": "chat", "text": ""}]}), 422)

# --- async job ----------------------------------------------------------------
pool = val + load("train")
job_rows = [pool[i % len(pool)] for i in range(JOB_SIZE)]
job_tickets = [as_request(r, f"JOB-{i:05d}") for i, r in enumerate(job_rows)]
t0 = time.time()
code, job, rh = call("POST", "/batch/jobs", {"tickets": job_tickets}, headers={"Idempotency-Key": f"smoke-{int(t0)}"})
expect(f"POST /batch/jobs ({JOB_SIZE} tickets) -> 202", (code, job, rh), 202, "batch_job_status")
check("202 came back within 5 s", time.time() - t0 < 5, f"{time.time() - t0:.1f}s")
job_id = (job or {}).get("job_id", "missing")
check("Location header points at the job", rh.get("Location") == f"/batch/jobs/{job_id}", str(rh.get("Location")))
code2, replay, _ = call("POST", "/batch/jobs", {"tickets": job_tickets}, headers={"Idempotency-Key": f"smoke-{int(t0)}"})
check("same Idempotency-Key returns the same job", code2 == 202 and (replay or {}).get("job_id") == job_id)

status = job or {}
polls = 0
while status.get("status") in ("queued", "running") and time.time() - t0 < 1800:
    time.sleep(2)
    code, status, rh = call("GET", f"/batch/jobs/{job_id}")
    polls += 1
    ok, why = valid("batch_job_status", status)
    if code != 200 or not ok:
        check("job status stays valid while polling", False, f"{code} {why}")
        break
    hc, _, _ = call("GET", "/health", key=None)
    if hc != 200:
        check("/health stays up while a job runs", False, str(hc))
elapsed = time.time() - t0
check(f"job succeeded ({elapsed:.0f}s, {polls} polls)", status.get("status") == "succeeded", str(status))

code, res, _ = call("GET", f"/batch/jobs/{job_id}/results")
expect("GET results -> 200", (code, res, None), 200, "batch_job_results")
preds = (res or {}).get("predictions", [])
check("results in request order", [p["ticket_id"] for p in preds] == [t["ticket_id"] for t in job_tickets])
code, page, _ = call("GET", f"/batch/jobs/{job_id}/results?offset=10&limit=20")
check("paging is stable", code == 200 and page["predictions"] == preds[10:30] and page["next_offset"] == 30)
check("job answers equal /predict/batch answers",
      [{k: v for k, v in p.items() if k != "ticket_id"} for p in preds[:100]]
      == [{k: v for k, v in p.items() if k != "ticket_id"} for p in (out or {}).get("predictions", [])][:100] if JOB_SIZE >= 100 else True)
expect("unknown job -> 404", call("GET", "/batch/jobs/00000000-0000-0000-0000-000000000000"), 404)
dcode, _, _ = call("DELETE", f"/batch/jobs/{job_id}")
check("DELETE job -> 204", dcode == 204, str(dcode))
expect("deleted job -> 404", call("GET", f"/batch/jobs/{job_id}"), 404)

# --- quick quality read-out (not a pass/fail check) ---------------------------
if preds:
    truth = job_rows[:len(preds)]
    acc = sum(p["category"] == t["category"] for p, t in zip(preds, truth)) / len(preds)
    urg = sum(p["is_urgent"] == t["is_urgent"] for p, t in zip(preds, truth)) / len(preds)
    print(f"\ninfo  agreement with dataset labels on these {len(preds)} tickets: category {acc:.1%}, urgency {urg:.1%}")
    print("      (the live model was trained on these files, so this only proves the pipeline works end to end)")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: " + "; ".join(failures)))
sys.exit(1 if failures else 0)
