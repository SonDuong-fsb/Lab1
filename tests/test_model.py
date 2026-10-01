import pytest

from app.config import DECLINE_THRESHOLD, REVIEW_THRESHOLD
from app.model import FEATURE_COLUMNS, CreditRiskModel
from tests.test_api import GOOD_APPLICANT


class TestDecide:
    """Threshold boundaries of the decision rule."""

    @pytest.mark.parametrize(
        "probability,band,decision",
        [
            (0.0, "LOW", "APPROVE"),
            (REVIEW_THRESHOLD - 0.0001, "LOW", "APPROVE"),
            (REVIEW_THRESHOLD, "MEDIUM", "REVIEW"),
            (DECLINE_THRESHOLD - 0.0001, "MEDIUM", "REVIEW"),
            (DECLINE_THRESHOLD, "HIGH", "DECLINE"),
            (1.0, "HIGH", "DECLINE"),
        ],
    )
    def test_boundaries(self, probability, band, decision):
        assert CreditRiskModel.decide(probability) == {
            "risk_band": band,
            "decision": decision,
        }


class TestToFrame:
    """Mapping from the API payload to the training layout."""

    def test_columns_match_training_order(self):
        frame = CreditRiskModel.to_frame([GOOD_APPLICANT])
        assert list(frame.columns) == FEATURE_COLUMNS

    def test_pay_status_skips_pay_1(self):
        applicant = dict(GOOD_APPLICANT, pay_status=[1, 2, 3, 4, 5, 6])
        row = CreditRiskModel.to_frame([applicant]).iloc[0]
        assert [row[c] for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] == [
            1, 2, 3, 4, 5, 6
        ]
        assert "PAY_1" not in FEATURE_COLUMNS

    def test_amounts_are_numbered_from_one(self):
        row = CreditRiskModel.to_frame([GOOD_APPLICANT]).iloc[0]
        assert row["BILL_AMT1"] == GOOD_APPLICANT["bill_amt"][0]
        assert row["BILL_AMT6"] == GOOD_APPLICANT["bill_amt"][5]
        assert row["PAY_AMT1"] == GOOD_APPLICANT["pay_amt"][0]
        assert row["PAY_AMT6"] == GOOD_APPLICANT["pay_amt"][5]


class TestLoading:
    """Artifact loading."""

    def test_missing_file_is_re_raised(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CreditRiskModel(model_path=str(tmp_path / "absent.joblib"))
