import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("Churn_Modelling.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. Remove Unnecessary Columns
# ==============================

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)


# ==============================
# 3. Separate Features and Target
# ==============================

X = df.drop("Exited", axis=1)
y = df["Exited"]


# ==============================
# 4. Encode Categorical Columns
# ==============================

X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True
)


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 6. Feature Scaling
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# 7. Create Models
# ==============================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=6
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
}


# ==============================
# 8. Train and Evaluate Models
# ==============================

results = []

best_model = None
best_f1 = 0
best_model_name = ""


for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))


    # Select best model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# ==============================
# 9. Model Comparison
# ==============================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
)

print("\n\nMODEL COMPARISON")
print("==============================")
print(results_df)


# ==============================
# 10. Save Best Model
# ==============================

joblib.dump(best_model, "best_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(X.columns.tolist(), "feature_columns.pkl")


print("\n==============================")
print("BEST MODEL")
print("==============================")

print("Model:", best_model_name)
print("F1-Score:", round(best_f1, 4))

print("\nBest model saved as: best_model.pkl")
print("Scaler saved as: scaler.pkl")
print("Feature columns saved as: feature_columns.pkl")

print("\nModel training completed successfully!")