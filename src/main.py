import logging

from data_loader import collect_cve_data
from preprocessing import preprocess_cve_data
from feature_extraction import fit_vectorizer, save_vectorizer
from train_model import train_models, save_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline():
    logging.info("Starting CVE pipeline")

    df_raw = collect_cve_data(total=1500, results_per_page=1000)
    if df_raw.empty:
        logging.error("No data loaded from NVD API; aborting")
        return

    df_clean, label_encoder = preprocess_cve_data(df_raw)

    X, vectorizer = fit_vectorizer(df_clean, text_column="description_clean", max_features=5000)
    y = df_clean["label"]

    clf_lr, clf_rf, metrics = train_models(X, y)

    save_vectorizer(vectorizer, "src/models/tfidf_vectorizer.joblib")
    save_model(label_encoder, "src/models/label_encoder.joblib")
    save_model(clf_rf, "src/models/random_forest_model.joblib")
    save_model(clf_lr, "src/models/logistic_regression_model.joblib")

    logging.info("Model results: %s", metrics)

    for model_name, model_metrics in metrics.items():
        logging.info("=== %s ===", model_name)
        for k, v in model_metrics.items():
            if k == "report":
                logging.info("\n%s", v)
            else:
                logging.info("%s: %.4f", k, v)


if __name__ == "__main__":
    run_pipeline()
