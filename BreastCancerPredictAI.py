"""
#!/usr/bin/python3
 # Created     : 2025-08-25 (Aug, Mon) 11:26:30 CEST
 Author(s)   : Mahbub Alam
 File        : BreastCancerPredictAI.py
 Created     : 2025-03-25
 Description : Breast Cancer Prediction using Machine Learning # {{{

This project explores the **Breast Cancer Wisconsin Diagnostic Dataset** to build a machine learning model that can classify tumors as *malignant* or *benign*.
The goal is to demonstrate end-to-end ML workflow:
- Data loading & cleaning
- Exploratory data analysis (EDA)
- Feature engineering & preprocessing
- Model training & evaluation
- Insights and conclusions

Such predictive modeling can support early detection and assist healthcare professionals, though models should **never replace medical diagnosis**.

# }}}
"""

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re

# ==================[[ Data Loading ]]==================={{{
print(f"")
print(68*"=")
print(f"==={22*'='}[[ Data Loading ]]{22*'='}===\n")
# ==========================================================

"""
We start by loading the cleaned breast cancer dataset.
Each row represents a tumor with various features extracted from digitized images of a fine needle aspirate (FNA).
The target column indicates whether the tumor is **malignant (cancerous)** or **benign (non-cancerous)**.

"""

bd = pd.read_csv('breast_cancer_data.csv')

# }}}

# ===============[[ Data Preprocessing ]]================{{{
print(f"")
print(68*"=")
print(f"==={19*'='}[[ Data Preprocessing ]]{19*'='}===\n")
# ==========================================================

"""
Machine learning algorithms require clean, numerical, and scaled data.
Steps:
- Handle missing values (if any)
- Encode target labels (Malignant = 1, Benign = 0)
- Normalize/standardize features
- Split dataset into training and testing sets

Looks like there is an extra comma at the end of the columns.
It's named "Unnamed: 32"
"""

print(bd.head())
print(bd.columns)

bd = bd.drop(columns="Unnamed: 32")
print(bd.columns)
bd.to_csv('breast_cancer_data_cleaned.csv', index=False)

# Data already cleaned, loading clean data
bd = pd.read_csv('breast_cancer_data_cleaned.csv')
print(bd.columns) # Output: ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# No missing data
missing_data = bd.isna().any()
print(missing_data.any())


# }}}

# ===============[[ Exploratory data analysis (EDA) ]]================{{{
print(f"")
print(68*"=")
print(f"==={19*'='}[[ Exploratory data analysis (EDA) ]]{19*'='}===\n")
# ==========================================================

"""
Understanding the dataset is key before building models.
We will:
- Check class distribution (malignant vs. benign)
- Visualize feature distributions
- Explore correlations between features
"""

print(bd.head())
print(bd.info())

# Proportion of benign and malignant tumors

print(bd['diagnosis'].value_counts())

# Creating binary labels for diagnosis
def lookup_diagnosis(label):
    if label == 'M':
        return 1
    else:
        return 0

X = bd.drop(columns="diagnosis")
y = pd.factorize(bd["diagnosis"], sort=True)[0]

print(X.head())
print(y[:5])


# }}}

# ==============[[ Model Training ]]==============={{{

"""
We will train multiple classifiers (e.g., Logistic Regression, Random Forest, SVM, etc.)
and compare their performance to find the most effective model.
Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
"""

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)

log_reg.fit(X, y)

# Finding the probabilities for class 1 (malignant)
probs = log_reg.predict_proba(X)[:,1]

# ===============[[ Output title like this ]]===============
print(f"")
print(68*"=")
print(f"==={8*'='}[[ Deciding a threshold for positive results ]]{8*'='}==\n")
# ==========================================================

"""
Given the above probs for each class (0 and 1) it might seem natural to choose 0.5
as a threshold for positive result, i.e., if probability for positive is >= 0.5
we might naively want to call that a positive result.

But a serious diagnosis such as breast cancer should have low threshold to avoid
high number of false negatives.
Also since the ML algorithms output will be reviewed by professional it is prudent
to keep the threshold low.

Below we check the precision recall curve to get some ideas about the threshold.

#### Definitions

P = (actual) positive data points

N = (actual) negative data points

TP = True positive  : a data point marked positive by the model that is actually positive

FP = False positive : a data point marked positive by the model that is actually negative

FN = False negative : a data point marked negative by the model that is actually positive

Precision = TP / (TP + FP) :
Of the points the model called positive, how many were truly positive?
Here TP + FP are the number of data points the model marked positive.

Recall = TP / (TP + FN) :
Of the truly positive points, how many did the model correctly find?
Here TP + FN are the number of data points that are actually positive.
"""

from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score
precision, recall, thresholds = precision_recall_curve(y, probs)

plt.plot(thresholds, precision[1:], label="Precision")
plt.plot(thresholds, recall[1:], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()


# ===============[[ Output title like this ]]===============
print(f"")
print(68*"=")
print(f"==={18*'='}[[ Choosing a threshold ]]{18*'='}===\n")
# ==========================================================

"""
Since this is a medical application, we don't want the model to miss positives.
We care about high recall, about 99%.
Let's find a threshold that achieves this.
"""

from sklearn.metrics import confusion_matrix

def pick_threshold_for_recall(y_true, probs, target_recall=0.99):
    """
    Returns the highest threshold whose recall >= target_recall.
    Using the highest such threshold usually gives better precision.
    """
    _, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall have length = len(thresholds)+1; align by dropping the first PR point
    recall_t = recall[1:]
    thresholds_t = thresholds

    # indices where recall constraint is satisfied
    ok = np.where(recall_t >= target_recall)[0]
    if len(ok) == 0:
        # cannot reach target recall; fall back to threshold=0 (max recall)
        chosen = 0.0
    else:
        # choose the largest threshold that still satisfies recall >= target
        chosen = thresholds_t[ok[-1]]
    return float(chosen)

threshold = pick_threshold_for_recall(y, probs, target_recall=.99)

# }}}

# ================[[ Model Evaluation ]]================={{{
print(f"")
print(68*"=")
print(f"==={20*'='}[[ Model Evaluation ]]{20*'='}===\n")
# ==========================================================

"""
We compare model performance on the test set.  
Metrics and confusion matrices help us assess:
- How well the model detects malignant tumors (sensitivity/recall)
- How well it avoids false positives (specificity/precision)

The ROC curve and AUC score further summarize predictive power.

"""

def metrics_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]]
    }


report = metrics_at_threshold(y, probs, threshold)

print(report)

# }}}

# ===================[[ Conclusion ]]===================={{{
print(f"")
print(68*"=")
print(f"==={23*'='}[[ Conclusion ]]{23*'='}===\n")
# ==========================================================

"""

- The models show strong ability to distinguish between malignant and benign tumors.
- [Insert best model name here] achieved the highest performance, with [XX%] accuracy and strong recall.
- This demonstrates the potential of ML in assisting medical diagnostics.

⚠️ Note: This project is for educational and demonstration purposes only.  
It should **not** be used for clinical decision-making.
"""


# }}}

# ===================[[ Next Steps ]]===================={{{
print(f"")
print(68*"=")
print(f"==={23*'='}[[ Next Steps ]]{23*'='}===\n")
# ==========================================================

"""
Potential improvements:
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Feature selection to reduce dimensionality
- Ensemble methods for better generalization
- Deployment as a simple web app (e.g., with Flask or Streamlit)

This would make the project even more practical and showcase end-to-end ML engineering skills.
"""


# }}}
