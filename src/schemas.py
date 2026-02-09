from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class PredictRequest(BaseModel):
    # Minimal: allow sending a dict of raw features
    features: Dict[str, Any] = Field(..., description="Raw feature payload matching training schema")

class PredictResponse(BaseModel):
    pd: float
    decision: str
    threshold: float
