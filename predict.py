import pickle
import pandas as pd
from pathlib import Path

_DIR = Path(__file__).parent

# Load model and encoder
with open(_DIR / "career_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

FEATURES = [
    "logical_reasoning", "numerical_ability", "verbal_ability",
    "creativity", "analytical_thinking", "leadership",
    "spatial_ability", "memory_retention", "empathy",
    "interest_physics", "interest_chemistry", "interest_biology",
    "interest_maths", "interest_literature", "interest_economics",
    "interest_arts", "interest_computers", "interest_history",
    "interest_psychology", "interest_business",
]

def predict_top3(student_scores: dict):
    """
    Takes a dict of feature scores and returns top 3 career recommendations.

    Example input:
    {
        "logical_reasoning": 88,
        "numerical_ability": 85,
        "verbal_ability": 55,
        ...
    }
    """
    input_vector = pd.DataFrame([{f: student_scores[f] for f in FEATURES}])

    # Get probabilities for each class
    probs = model.predict_proba(input_vector)[0]
    career_labels = le.classes_

    # Rank careers by probability
    ranked = sorted(zip(career_labels, probs), key=lambda x: x[1], reverse=True)

    print("\n🎯 Top 3 Career Recommendations:")
    print("-" * 40)
    for i, (career, prob) in enumerate(ranked[:3], 1):
        print(f"  {i}. {career.replace('_', ' ')} — Confidence: {prob * 100:.1f}%")
    print("-" * 40)

    return ranked[:3]


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample student profile (engineering-leaning)
    sample_student = {
        "logical_reasoning":   90,
        "numerical_ability":   88,
        "verbal_ability":      52,
        "creativity":          58,
        "analytical_thinking": 87,
        "leadership":          62,
        "interest_physics":    91,
        "interest_chemistry":  74,
        "interest_biology":    42,
        "interest_maths":      93,
        "interest_literature": 38,
        "interest_economics":  50,
        "interest_arts":       33,
    }

    predict_top3(sample_student)

    print("\n--- Testing an Arts-leaning student ---")
    arts_student = {
        "logical_reasoning":   52,
        "numerical_ability":   45,
        "verbal_ability":      91,
        "creativity":          90,
        "analytical_thinking": 63,
        "leadership":          70,
        "interest_physics":    33,
        "interest_chemistry":  30,
        "interest_biology":    38,
        "interest_maths":      40,
        "interest_literature": 94,
        "interest_economics":  58,
        "interest_arts":       92,
    }

    predict_top3(arts_student)
