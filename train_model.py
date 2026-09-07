import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_and_train():
    try:
        df = pd.read_csv("data/spam.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("data/spam.csv", encoding="latin-1")

    print("Raw Dataset Size:", len(df))

    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    df = df.dropna()
    df = df.drop_duplicates()

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]

    X = df["message"]

    y = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)

    model.fit(
        X_train_vec,
        y_train
    )

    y_pred = model.predict(X_test_vec)

    print("\nClean Dataset Size:", len(df))
    print("Ham Messages:", (y == 0).sum())
    print("Spam Messages:", (y == 1).sum())
    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Ham", "Spam"]
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    os.makedirs("model", exist_ok=True)

    joblib.dump(
        model,
        "model/spam_model.pkl"
    )

    joblib.dump(
        vectorizer,
        "model/vectorizer.pkl"
    )

    print("\n✅ Model and vectorizer saved successfully.")

if __name__ == "__main__":
    generate_and_train()