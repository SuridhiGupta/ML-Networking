import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_models(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf_lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    clf_rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=random_state, n_jobs=-1)

    logging.info("Training Logistic Regression")
    clf_lr.fit(X_train, y_train)

    logging.info("Training Random Forest")
    clf_rf.fit(X_train, y_train)

    results = {}
    for name, model in [("LogisticRegression", clf_lr), ("RandomForest", clf_rf)]:
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "report": classification_report(y_test, y_pred, zero_division=0),
        }

    return clf_lr, clf_rf, results


def save_model(model, path):
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dst)
    logging.info("Saved model to %s", dst)


def load_model(path):
    return joblib.load(path)
