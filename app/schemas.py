from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

PAY_STATUS_MIN = -2
PAY_STATUS_MAX = 8


class CreditApplication(BaseModel):
    """One applicant, as the core banking system sends it."""

    limit_bal: float = Field(
        ..., gt=0, le=2_000_000, description="Credit limit in NT dollars", examples=[120000]
    )
    sex: Literal[1, 2] = Field(..., description="1 = male, 2 = female", examples=[2])
    education: Literal[1, 2, 3, 4] = Field(
        ..., description="1 = graduate school, 2 = university, 3 = high school, 4 = others",
        examples=[2],
    )
    marriage: Literal[1, 2, 3] = Field(
        ..., description="1 = married, 2 = single, 3 = others", examples=[2]
    )
    age: int = Field(..., ge=18, le=100, description="Age in years", examples=[34])
    pay_status: List[int] = Field(
        ...,
        min_length=6,
        max_length=6,
        description=(
            "Repayment status for months t-1 .. t-6. "
            "-2 = no consumption, -1 = paid in full, 0 = revolving credit, "
            "1..8 = months of payment delay."
        ),
        examples=[[0, 0, 0, 0, 0, 0]],
    )
    bill_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Statement amount for months t-1 .. t-6",
        examples=[[12000, 11500, 11000, 10500, 10000, 9500]],
    )
    pay_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Amount actually paid for months t-1 .. t-6",
        examples=[[1500, 1500, 1500, 1500, 1500, 1500]],
    )

    @field_validator("pay_status")
    @classmethod
    def validate_pay_status(cls, values: List[int]) -> List[int]:
        if any(not PAY_STATUS_MIN <= v <= PAY_STATUS_MAX for v in values):
            raise ValueError(
                f"pay_status values must be between {PAY_STATUS_MIN} and {PAY_STATUS_MAX}"
            )
        return values

    @field_validator("pay_amt")
    @classmethod
    def validate_pay_amt(cls, values: List[float]) -> List[float]:
        if any(v < 0 for v in values):
            raise ValueError("pay_amt values must not be negative")
        return values


class PredictionResponse(BaseModel):
    """Scoring result plus the decision derived from it."""

    default_probability: float = Field(..., ge=0.0, le=1.0, examples=[0.1234])
    risk_band: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., examples=["LOW"])
    decision: Literal["APPROVE", "REVIEW", "DECLINE"] = Field(..., examples=["APPROVE"])
    review_threshold: float = Field(..., examples=[0.30])
    decline_threshold: float = Field(..., examples=[0.60])
    model_version: str = Field(..., examples=["1.0.0"])


class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Up to 500 applications scored in one call."""

    applications: List[CreditApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the request."""

    predictions: List[PredictionResponse]
    total_count: int
