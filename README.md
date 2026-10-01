# Credit Default Risk Scoring API

FastAPI service that scores a credit card applicant's probability of defaulting on the next payment and maps it to an underwriting decision.

## Endpoints

| Method | Path             | Purpose                                              |
|--------|------------------|------------------------------------------------------|
| GET    | `/health`        | Readiness: status and whether the model is loaded    |
| POST   | `/predict`       | Score one applicant                                  |
| POST   | `/predict/batch` | Score 1 to 500 applicants in a single model call     |
| GET    | `/model/info`    | Model version, training metrics, thresholds in force |
| GET    | `/docs`          | Swagger UI                                           |

Status codes: `200` scored, `422` request rejected by the schema, `503` model not loaded, `500` unexpected scoring error.

## Decision rule

| Probability            | Risk band | Decision |
|------------------------|-----------|----------|
| `< 0.30`               | LOW       | APPROVE  |
| `0.30` to `< 0.60`     | MEDIUM    | REVIEW   |
| `>= 0.60`              | HIGH      | DECLINE  |

Thresholds come from `REVIEW_THRESHOLD` and `DECLINE_THRESHOLD`.

## Configuration

| Variable            | Default                      |
|---------------------|------------------------------|
| `MODEL_PATH`        | `models/credit_model.joblib` |
| `MODEL_VERSION`     | `1.0.0`                      |
| `REVIEW_THRESHOLD`  | `0.30`                       |
| `DECLINE_THRESHOLD` | `0.60`                       |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

Python 3.11 or 3.13 is recommended; the pinned numpy, pandas and scikit-learn versions do not ship wheels for 3.14.

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"limit_bal":300000,"sex":2,"education":1,"marriage":2,"age":38,
       "pay_status":[-1,-1,-1,-1,-1,-1],
       "bill_amt":[12000,11500,11000,10500,10000,9500],
       "pay_amt":[12000,11500,11000,10500,10000,9500]}'
```

## Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Docker

The image does not contain the trained model. Train it on the host first, then Compose mounts `./models` read-only into the container.

```bash
python scripts/train_model.py
docker compose up --build
docker inspect --format='{{.State.Health.Status}}' credit-risk-api
```

The container runs as a non-root user and its health check requires `model_loaded` to be `true`.
