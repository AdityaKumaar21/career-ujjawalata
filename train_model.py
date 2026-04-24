import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("student_data.csv")

FEATURES = [
    "logical_reasoning", "numerical_ability", "verbal_ability",
    "creativity", "analytical_thinking", "leadership",
    "spatial_ability", "memory_retention", "empathy",
    "interest_physics", "interest_chemistry", "interest_biology",
    "interest_maths", "interest_literature", "interest_economics",
    "interest_arts", "interest_computers", "interest_history",
    "interest_psychology", "interest_business",
]

X = df[FEATURES]
y = df["career_path"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder
with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ Model saved as career_model.pkl")
print("✅ Label encoder saved as label_encoder.pkl")
