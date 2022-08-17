

y_true = ["Beatles"]*2 + ["Eminem"]*2
X_test = "We love a loyal submarine"

CORPUS = [
    "we all love a yellow submarine",             # Beatles
    "yesterday, my submarine was in love",        # Beatles
    "we are love trouble with loyalty here",      # Eminem
    "loyalty to us is worth more than love is"    # Eminem
]

def train_model(X_train, y_train):
    """Return a NB classifier fit on a selected dataset (CORPUS) consisting of a list of strings."""
    m = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    fitted_model = m.fit(X_train, y_train)
    return fitted_model

def make_predictions(X_test, fitted_model):
    """Takes a fitted NB classifier and returns predictions (hard and soft)."""
    X_test = [X_test]
    predictions = fitted_model.predict(X_test) # "hard"-predictions
    probs = fitted_model.predict_proba(X_test) # "soft"-predictions
    return predictions, probs



if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    import pickle

    NB_clf = train_model(
        X_train = CORPUS,
        y_train = y_true
    )

with open("naive_classifier.bin", "wb") as file:
    pickle.dump(NB_clf, file)

