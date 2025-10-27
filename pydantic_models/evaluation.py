from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

class MetricScale(str, Enum):
    LIKERT_7 = "likert-7"
    LIKERT_5 = "likert-5"
    LIKERT_3 = "likert-3"
    BINARY = "binary"
    COMPARISON = "comparison"

    

class BaseEvaluationResponse(BaseModel):
    reasoning: str = Field(..., title="Brief reasoning for the evaluation.")
    confidence: int = Field(..., ge=1, le=10, title="Confidence level between 1 and 10")

class BinaryEvaluationResponse(BaseEvaluationResponse):
    """Response schema for binary evaluation (yes/no)"""
    decision: Literal["yes", "no"] = Field(..., title="The binary decision")

class Likert7EvaluationResponse(BaseEvaluationResponse):
    """Response schema for Likert scale evaluation"""
    rating: int = Field(..., ge=1, le=7, title="Rating on the 7-point scale")


class Likert5EvaluationResponse(BaseEvaluationResponse):
    """Response schema for 5-point Likert scale evaluation"""
    rating: int = Field(..., ge=1, le=5, title="Rating on the 5-point scale")

class Likert3EvaluationResponse(BaseEvaluationResponse):
    """Response schema for 3-point Likert scale evaluation"""
    rating: int = Field(..., ge=1, le=3, title="Rating on the 3-point scale")

class ComparisonEvaluationResponse(BaseEvaluationResponse):
    """Response schema for comparison evaluation"""
    decision: Literal["A", "B"] = Field(..., title="The better option.")