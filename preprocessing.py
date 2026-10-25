import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

print("Original Dataset Shape:")
print(df.shape)

# Remove unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

print("\nAfter Removing Unnecessary Columns:")
print(df.shape)

# Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)

print("\nFeatures after Encoding:")
print(X.columns.tolist())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing completed successfully!")