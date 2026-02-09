from pathlib import Path
import joblib
import pandas as pd

MODELS_DIR = Path("models")

# Cached singletons
_model = None
_feature_list = None
_segment_artifacts = None

def load_artifacts():
    """Carica gli artefatti necessari per predizione.
    - model.joblib: modello calibrato globale (CalibratedClassifierCV)
    - feature_list.csv: lista feature attese dall'API
    - segment_calibrators.joblib: (opzionale) calibrazione per segmenti merchant
    """
    global _model, _feature_list, _segment_artifacts
    if _model is None:
        _model = joblib.load(MODELS_DIR / "model.joblib")
    if _feature_list is None:
        fl = pd.read_csv(MODELS_DIR / "feature_list.csv")
        _feature_list = fl["feature"].tolist()
    if _segment_artifacts is None:
        path = MODELS_DIR / "segment_calibrators.joblib"
        _segment_artifacts = joblib.load(path) if path.exists() else None
    return _model, _feature_list, _segment_artifacts

def _apply_segment_calibration_if_available(pd_hat: float, payload: dict, seg_artifacts) -> float:
    """Se disponibile, applica una calibrazione isotonic specifica per merchant.
    Nota: per semplicitÃ  in API, gestiamo 1 record alla volta.
    """
    if seg_artifacts is None:
        return pd_hat

    merchant_col = seg_artifacts.get("merchant_col")
    if not merchant_col:
        return pd_hat

    merchant_value = payload.get(merchant_col, None)
    if merchant_value is None:
        merchant_value = "Other"

    kept = set(seg_artifacts.get("kept_merchants", []))
    other_label = seg_artifacts.get("other_label", "Other")
    seg = str(merchant_value) if str(merchant_value) in kept else other_label

    calibrators = seg_artifacts.get("calibrators", {})
    fallback = seg_artifacts.get("fallback_global", None)

    iso = calibrators.get(seg, fallback)
    if iso is None:
        return pd_hat

    # isotonic expects array-like
    return float(iso.transform([pd_hat])[0])

def predict_pd(payload: dict) -> float:
    model, feature_list, seg_artifacts = load_artifacts()

    # Build 1-row dataframe with all expected columns
    row = {k: payload.get(k, None) for k in feature_list}
    X = pd.DataFrame([row], columns=feature_list)

    pd_hat = float(model.predict_proba(X)[:, 1][0])
    pd_hat = _apply_segment_calibration_if_available(pd_hat, payload, seg_artifacts)
    return pd_hat
