from fastapi import FastAPI
from src.schemas import PredictRequest, PredictResponse
from src.predict import predict_pd

app = FastAPI(title="Klarna PD API", version="1.0")

# Choose a default underwriting threshold (example)
DEFAULT_THRESHOLD = 0.10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, threshold: float = DEFAULT_THRESHOLD):
    pd_hat = predict_pd(req.features)
    decision = "approve" if pd_hat <= threshold else "decline"
    return PredictResponse(pd=pd_hat, decision=decision, threshold=float(threshold))
