import re
import joblib
import logging
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

regex_pattern = re.compile(r"[^a-zA-Z\s]")
stop_words_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
pos_mapping = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
model_dir = "../model/ml_model.joblib"


def _get_wordnet_pos(tag):
    return pos_mapping.get(tag[0], wordnet.NOUN)


def _preprocess_text(text):
    text = text.lower()
    text = regex_pattern.sub("", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set]
    pos_tags = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    return " ".join(tokens)


def preprocess_data():
    logging.info(f"Preprocessing data ...")
    # Read the dataset
    df = pd.read_csv("../data/Britannia.csv")

    # Preprocess the dataset
    negative_reviews = df.loc[df["Negative_Review"] != "No Negative", "Negative_Review"]
    positive_reviews = df.loc[df["Positive_Review"] != "No Positive", "Positive_Review"]

    review_df = pd.DataFrame(
        {
            "Review": pd.concat(
                [negative_reviews, positive_reviews], ignore_index=True
            ),
            "Sentiment": ["Negative"] * len(negative_reviews)
            + ["Positive"] * len(positive_reviews),
        }
    )

    review_df["Cleaned_Review"] = review_df["Review"].apply(_preprocess_text)

    # Split the dataset into training and testing sets
    return train_test_split(
        review_df[["Review", "Cleaned_Review"]],
        review_df["Sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=review_df["Sentiment"],
    )


def _perform_grid_search(name, clf, param_grid, X_train, y_train):
    """Perform GridSearchCV on a classifier with given parameters."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train["Cleaned_Review"], y_train)
    return (
        grid_search.best_score_,
        grid_search.best_estimator_,
        grid_search.best_params_,
    )


def train(X_train, y_train):
    # Define classifiers and their hyperparameters
    classifiers = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Naive Bayes": (MultinomialNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "Decision Tree": (DecisionTreeClassifier(), {"clf__max_depth": [10, 50, 100]}),
        "SVM": (SVC(), {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
        "Random Forest": (
            RandomForestClassifier(),
            {"clf__n_estimators": [10, 50, 100]},
        ),
    }

    # Perform GridSearchCV to find the best classifier
    best_score, best_model, best_params = 0, None, None

    for name, (clf, param_grid) in classifiers.items():
        logging.info(f"Training {name} {param_grid} ...")
        score, model, params = _perform_grid_search(
            name, clf, param_grid, X_train, y_train
        )
        if score > best_score:
            best_score, best_model, best_params = score, model, params

    # Save the best model
    joblib.dump(best_model, model_dir)
    logging.info(
        f"Best model ({best_model.named_steps['clf'].__class__.__name__} {best_params}: {best_score}) saved as {model_dir}"
    )

    return best_score, best_model, best_params


def evaluate(X_test, y_test):
    # Load the best model
    model = joblib.load(model_dir)

    # Evaluate the model
    y_pred = model.predict(X_test["Cleaned_Review"])
    report = classification_report(y_test, y_pred, digits=4)
    logging.info(f"Classification report:\n{report}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    best_score, best_model, best_params = train(X_train, y_train)
    evaluate(X_test, y_test)
