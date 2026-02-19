"""
train_model.py

Goal:
- Load CEFR-labeled Spanish texts (UniversalCEFR)
- Extract features with spaCy (via features.extractor)
- Train a classifier to predict CEFR level (1-6)
- Save trained model + metadata for serving
"""


import os
# 1. Imports
# - import numpy
# - import sklearn pieces (RandomForestClassifier, metrics)
# - import joblib for saving model
# - from datasets import load_dataset
# - from features.extractor import extract_features
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib 
from features.extractor import extract_features
from datasets import load_dataset

# 2. Load UniversalCEFR dataset
# - call load_dataset("UniversalCEFR/cefr_sp_en")
# - split into train/validation/test

ds = load_dataset("UniversalCEFR/hablacultura_es")
full = ds["train"]

examples = [full[i] for i in range(len(full))]
# for i, doc in enumerate(examples):
#     print(list(doc.keys()))
# def filter_cefr(docs):


# 2) Extract texts and labels
texts = [ex["text"] for ex in examples]
labels_cefr = [ex["cefr_level"] for ex in examples]

CEFR_TO_INT = {"A1": 1, 
               "A2": 1, 
               "B": 2,
               "B1": 2, 
               "B2": 2, 
               "C": 3,
               "C1": 3, 
               "C2": 3}
labels = [CEFR_TO_INT[l] for l in labels_cefr]

# 3) Train / temp split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. Filter to Spanish texts only
# - define filter_spanish(example) -> example["lang"] == "es"
# - apply .filter(...) to each split

# def map_label(data, labels):
#     for i, doc in enumerate(data):
#         doc["label_int"] = labels[i]

# map_label(X_train, y_train)



# 4. Map CEFR labels ("A1"–"C2") to integers 1–6
# - define CEFR_TO_INT dict
# - define map_label(example) that adds example["label_int"]
# - apply .map(...) and filter out None labels


# 5. Convert dataset splits into feature matrices
# - define dataset_to_matrix(ds):
#     - loop over examples
#     - call extract_features(example["text"])
#     - collect feature values into X
#     - collect label_int into y
#     - capture feature_names from keys of last feature dict
# - call for train/valid/test → X_train, y_train, etc.

def ds_to_matrix(texts, labels):
    X = []
    for text in texts:
        feats = extract_features(text)
        X.append(list(feats.values()))
    feature_names = list(feats.keys()) if len(texts) > 0 else []
    X = np.array(X, dtype="float32")
    y = np.array(labels, dtype="int64")
    return X, y, feature_names

X_train_feats, y_train_arr, feature_names = ds_to_matrix(X_train, y_train)
X_test_feats,  y_test_arr,  _ = ds_to_matrix(X_test,  y_test)



# 6. Define and train classifier
# - instantiate RandomForestClassifier (or other model)
# - fit on X_train, y_train
# - optionally tune hyperparameters later

clf = RandomForestClassifier(
    n_estimators=300,   # a bit larger than default for stability
    max_depth=None,    # let trees grow until pure / min_samples constraints
    min_samples_leaf=2,  # small regularization to avoid tiny leaves
    n_jobs=-1,         # use all CPU cores
    random_state=42,   # reproducibility
)

# 7. Evaluate on validation and test sets
# - predict on X_valid, X_test
# - print classification_report for each

clf.fit(X_train_feats, y_train_arr)

y_pred_test = clf.predict(X_test_feats)
print(metrics.classification_report(y_test, y_pred_test))





# 8. Save trained model and feature names
# - create dict {"model": clf, "feature_names": feature_names}
# - joblib.dump(...) to "models/difficulty_model.pkl"


# 9. (Optional) Small CLI entry point
# - if __name__ == "__main__":
#     - run all the above steps
#     - print "Done" when finished
