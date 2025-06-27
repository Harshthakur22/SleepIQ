import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "wearable_tech_sleep_quality.csv"  # Update with correct path
df = pd.read_csv(file_path)

# Categorizing Sleep Quality Score
def categorize_sleep_quality(score):
    if score <= 4:
        return 0  # Poor Sleep
    elif 5 <= score <= 7:
        return 1  # Fair Sleep
    else:
        return 2  # Good Sleep

df["Sleep_Quality_Label"] = df["Sleep_Quality_Score"].apply(categorize_sleep_quality)

# Drop original Sleep Quality Score column
X = df.drop(columns=["Sleep_Quality_Score", "Sleep_Quality_Label"])
y = df["Sleep_Quality_Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance dataset using SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000)
}

# Train and evaluate models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Track the best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = (name, model)

# Display best model
print(f"\nBest Model: {best_model[0]} with Accuracy: {best_accuracy:.4f}")

# Confusion Matrix for best model
y_pred_best = best_model[1].predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Poor', 'Fair', 'Good'], yticklabels=['Poor', 'Fair', 'Good'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model[0]}")
plt.show()