---
title: Data Pulse
emoji: 🚦
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Data Pulse: RideEat support ticket classifier

TensorForge 2.0, Phase 2 (MVP). A service that reads a customer support ticket written in English,
Sinhala, Tamil, Singlish or Tanglish and returns the issue category, an optional second category,
the team to route it to, and whether it is urgent.

The block of settings at the very top of this file is read by Hugging Face Spaces (it tells the
Space to build the Dockerfile and expose port 8000). GitHub shows it as a small table; leave it there.

## Run it

```bash
docker build -t datapulse .
docker run -p 8000:8000 -e API_KEY=<your-team-key> datapulse
```

Then open http://localhost:8000 for the demo site, or call the API:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: <your-team-key>" -H "Content-Type: application/json" \
  -d '{"ticket_id":"T-1","channel":"chat","subject":"","text":"order eka late, refund ekak denna"}'
```

The container needs no internet access at runtime: code and model weights are inside the image.
The API key is read from the `API_KEY` environment variable at start-up and is never stored in the
repository or the image. If `API_KEY` is missing, every protected endpoint answers 401.

## What is in the box

| Deliverable | Where |
|---|---|
| Demo site and dashboard | `GET /` (served by the same container): route one ticket with matched keywords and a suggested first reply, a live inbox that sorts incoming tickets into team lanes, route a file as a background job, model results |
| Batch inference endpoint | `POST /predict`, `POST /predict/batch`, `POST /batch/jobs` and friends, see below |
| Training code | `training/train.py` (one command reproduces every number in this file) |
| Model artifact | `models/datapulse-1.0.0/` (`model.joblib`, `metrics.json`, `model_card.json`) |
| Contract tests | `tests/test_contract.py` (58 tests, validated against the organisers' JSON Schemas) |
| Live smoke test | `scripts/smoke_test.py <url> <key>` (35 checks against a running container or the hosted URL) |
| Report and video outline | `docs/REPORT.md`, `docs/VIDEO_SCRIPT.md` |

## API

Follows `specs/tensorforge-phase2-openapi-v2.yaml`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Public. 200 once the model is loaded, 503 while loading |
| `POST /predict` | One ticket |
| `POST /predict/batch` | 1 to 100 tickets, synchronous, atomic, order preserved |
| `POST /batch/jobs` | 1 to 5,000 tickets as a background job (202 + `Location` + `Retry-After`) |
| `GET /batch/jobs/{job_id}` | Status and progress |
| `GET /batch/jobs/{job_id}/results` | Results once succeeded, with `offset` / `limit` paging |
| `DELETE /batch/jobs/{job_id}` | Cancel or discard |

Behaviour worth knowing:

- **Auth**: `X-API-Key: <key>` or `Authorization: Bearer <key>`, compared in constant time, checked before the body is parsed.
- **Order of checks**: auth (401), content type (415), size (413), JSON parse (400), validation (422). Unknown routes and wrong methods answer JSON 404 / 405. Invalid input never produces a 5xx.
- **Consistency rules** are enforced in one place (`app/model.py`): `team` always follows `category`, `secondary_category` never equals `category`, spam is never urgent and never has a secondary label.
- **Jobs** are stored in SQLite (`DATA_DIR/jobs.sqlite3`), processed by one background thread in chunks of 100. Capacity is 1 running + 3 queued (then 429). `Idempotency-Key` returns the existing job. A job that was running when the container restarted is reported as `failed` with `error.code = "interrupted"`; queued jobs are picked up again. Finished jobs are kept 24 hours (`JOB_RETENTION_SECONDS`, minimum 6 hours), then answer 410.
- **`needs_human_review`** is `true` when `confidence < 0.60`. On validation that threshold routes 86.8% of tickets automatically at 95.7% accuracy; the flagged tickets are only 38% accurate, so they are the right ones to hand to a person.

Measured on a 2-vCPU machine: `/predict` p95 10 ms, 100-ticket batch 0.2 s, 5,000-ticket job 12 s, about 230 MB of memory.

## The model

No LLM is called anywhere. The classifier is three linear SVMs (category, urgency, secondary category)
on one shared feature matrix:

1. word TF-IDF
2. character n-gram TF-IDF (2 to 5 characters), robust to typos and spelling variation
3. character n-gram TF-IDF on **romanised** text: Sinhala and Tamil letters are mapped to Latin
   (`app/romanize.py`), so `සල්ලි කැපිලා` and the Singlish `salli kapila` produce the same features
4. a **domain concept lexicon** (`app/lexicon.py`): 37 concepts such as refund, delay, lost item, threat,
   app failure, each with keyword stems in English, romanised Sinhala/Singlish and romanised Tamil/Tanglish
5. the same lexicon applied after **boilerplate masking**: word trigrams that occur across many
   categories ("I have kept a copy of the confirmation") are learned from the data and removed first

Confidence is a softmax over the SVM margins with a temperature fitted on validation
(expected calibration error 0.594 before, 0.034 after).

### How it was validated and improved

The key finding: validation tickets describe the same kinds of problems as training tickets but with
**different wording**. Cross-validation inside the training file is about 98% for every model and says
nothing useful. Fitting on train and scoring on validation is the honest test, and it is what drove
every decision below.

| # | Model | Validation accuracy | Macro F1 | CV inside train |
|---|---|---|---|---|
| E1 | Baseline: word TF-IDF + logistic regression | 60.4% | 0.634 | 97.7% |
| E2 | Word TF-IDF + linear SVM | 64.0% | 0.665 | 98.1% |
| E3 | + character n-grams | 66.0% | 0.686 | 98.7% |
| E4 | + romanised character n-grams | 68.0% | 0.703 | 98.9% |
| E5 | + domain concept lexicon | 85.5% | 0.853 | 99.0% |
| E6 | + boilerplate-masked lexicon (final recipe) | 88.1% | 0.878 | 98.9% |

Final recipe on validation (fitted on train only): category accuracy **88.0%**, macro F1 **0.876**;
urgency F1 **0.924** (precision 0.936, recall 0.913); secondary category exact-label F1 **0.882**;
by language: Singlish 98%, English 88%, Sinhala 85%, Tamil 82%, mixed 80%, Tanglish 60% (20 tickets).

The shipped model is the same recipe refitted on train + validation (4,800 tickets) plus 30
hand-written spam examples (`training/extra_examples.py`).

**Read these numbers with care.** The lexicon was refined after looking at English validation errors,
so 88% is optimistic for wording the lexicon has never met. A fairer expectation on a fresh test set is
somewhere between E4 and E6. See `docs/REPORT.md` for this and the other limitations.

### Reproduce

```bash
pip install -r requirements-dev.txt      # Python 3.12 or 3.13
python -m training.train                 # about 3 minutes: prints the table above, writes models/datapulse-1.0.0/
pytest -q                                # 58 contract tests
python scripts/smoke_test.py http://localhost:8000 <your-team-key>   # against a running container
```

## Repository layout

```
app/            service code (main.py API, model.py, features.py, romanize.py, lexicon.py, jobs.py, validation.py, static/)
training/       train.py (experiments + final model), extra_examples.py
models/         versioned model artifacts; models/CURRENT names the one that is served
data/           organisers' train / validation files and DATA_NOTES.md
specs/          organisers' OpenAPI contract and JSON Schemas
tests/          contract tests
scripts/        smoke_test.py for a live URL
docs/           report and video outline
Dockerfile, requirements.txt, requirements-dev.txt, .env.example
```

## Hosting

Any platform that runs a Docker container works. On Hugging Face Spaces: create a Space with the
Docker SDK, push this repository to it, and add `API_KEY` under Settings, Variables and secrets,
as a **secret**. Free Spaces do not keep their disk across restarts, so job history is lost if the
Space restarts; mount a persistent volume at `DATA_DIR` on platforms that offer one.
