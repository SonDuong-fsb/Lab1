import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import DECLINE_THRESHOLD, MODEL_PATH, MODEL_VERSION, REVIEW_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_STATUS_COLUMNS
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


class CreditRiskModel:
    """Loads the trained pipeline and turns applications into decisions."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained pipeline from disk."""
        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["pipeline"]
            self.metadata = bundle.get("metadata", {})
            logger.info("Model loaded from %s", self.model_path)
        except FileNotFoundError:
            logger.error(
                "Model file not found at %s. Run 'python scripts/train_model.py' to create it.",
                self.model_path,
            )
            raise

    @staticmethod
    def to_frame(applications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten API payloads into the wide frame the model was trained on."""
        rows = []
        for application in applications:
            row = {
                "LIMIT_BAL": application["limit_bal"],
                "SEX": application["sex"],
                "EDUCATION": application["education"],
                "MARRIAGE": application["marriage"],
                "AGE": application["age"],
            }
            for name, value in zip(PAY_STATUS_COLUMNS, application["pay_status"]):
                row[name] = value
            for i, (bill, paid) in enumerate(
                zip(application["bill_amt"], application["pay_amt"]), start=1
            ):
                row[f"BILL_AMT{i}"] = bill
                row[f"PAY_AMT{i}"] = paid
            rows.append(row)
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    def predict_proba(self, applications: List[Dict[str, Any]]) -> np.ndarray:
        """Return the probability of default for each application."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        frame = self.to_frame(applications)
        return self.model.predict_proba(frame)[:, 1]

    @staticmethod
    def decide(probability: float) -> Dict[str, str]:
        """Turn a probability into a risk band and an underwriting decision."""
        if probability >= DECLINE_THRESHOLD:
            return {"risk_band": "HIGH", "decision": "DECLINE"}
        if probability >= REVIEW_THRESHOLD:
            return {"risk_band": "MEDIUM", "decision": "REVIEW"}
        return {"risk_band": "LOW", "decision": "APPROVE"}

    def score(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Score one application and return the full response payload."""
        probability = float(self.predict_proba([application])[0])
        result = {
            "default_probability": round(probability, 4),
            "review_threshold": REVIEW_THRESHOLD,
            "decline_threshold": DECLINE_THRESHOLD,
            "model_version": MODEL_VERSION,
        }
        result.update(self.decide(probability))
        return result

    def score_batch(self, applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score many applications in a single vectorised pass."""
        probabilities = self.predict_proba(applications)
        results = []
        for probability in probabilities:
            probability = float(probability)
            result = {
                "default_probability": round(probability, 4),
                "review_threshold": REVIEW_THRESHOLD,
                "decline_threshold": DECLINE_THRESHOLD,
                "model_version": MODEL_VERSION,
            }
            result.update(self.decide(probability))
            results.append(result)
        return results

    def is_loaded(self) -> bool:
        """Check whether the model is ready to serve."""
        return self.model is not None


_model_instance: Optional[CreditRiskModel] = None


def get_model() -> CreditRiskModel:
    """Get or create the model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = CreditRiskModel()
    return _model_instance
