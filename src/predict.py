import numpy as np

from feature_extraction import load_vectorizer
from train_model import load_model


RISK_MAP = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Critical",
    4: "Unknown",
}


def predict_risk(text, vectorizer, model, label_encoder):
    text_clean = text if isinstance(text, str) else ""
    x = vectorizer.transform([text_clean])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None

    label = label_encoder.inverse_transform([pred])[0] if label_encoder is not None else RISK_MAP.get(pred, "Unknown")

    explanation = {
        "predicted": label,
        "probabilities": {label_encoder.inverse_transform([i])[0] if label_encoder and i < len(label_encoder.classes_) else RISK_MAP.get(i, str(i)): float(p)
                          for i, p in enumerate(proba)} if proba is not None else None,
    }

    return explanation
