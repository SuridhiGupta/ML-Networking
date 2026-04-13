import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features=5000):
    return TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_features,
        min_df=3,
        max_df=0.85,
    )


def fit_vectorizer(df, text_column="description_clean", max_features=5000):
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[text_column].astype(str))
    return X, vectorizer


def transform_text(vectorizer, text_series):
    return vectorizer.transform(text_series.astype(str))


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)
