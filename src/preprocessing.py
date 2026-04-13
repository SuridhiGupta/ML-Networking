import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URL-like tokens using string matching only
    tokens = text.split()
    tokens = [tok for tok in tokens if tok.find("http://") == -1 and tok.find("https://") == -1]
    text = " ".join(tokens)

    # Keep only alphanumeric and whitespace
    filtered_chars = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            filtered_chars.append(ch)
        else:
            filtered_chars.append(" ")
    text = "".join(filtered_chars)

    # Normalize whitespace without regex
    text = " ".join(text.split()).strip()
    return text


def preprocess_cve_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["cve_id", "description", "risk_level"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataframe must include columns: {required_cols}")

    df = df.copy()
    df["description_clean"] = df["description"].apply(clean_text)
    df["risk_level"] = df["risk_level"].fillna("Unknown")

    df = df[df["description_clean"].str.len() > 20].reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["risk_level"])

    return df, label_encoder
