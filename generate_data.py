import numpy as np
import pandas as pd
import math

np.random.seed(42)

LIKERT_VALS = [10, 30, 50, 70, 90]
_BOUNDS = [20, 40, 60, 80]

def _norm_cdf(x, mean, std):
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2))))

def likert_probs(mean, std):
    cdf = [_norm_cdf(b, mean, std) for b in _BOUNDS]
    edges = [0.0] + cdf + [1.0]
    probs = [max(edges[i+1] - edges[i], 1e-9) for i in range(5)]
    total = sum(probs)
    return [p / total for p in probs]

def likert_sample(mean, std):
    return np.random.choice(LIKERT_VALS, p=likert_probs(mean, std))

# Each feature: (mean, std) — scores out of 100
ARCHETYPES = {
    "Engineering": {
        "logical_reasoning":   (90, 5),
        "numerical_ability":   (88, 5),
        "verbal_ability":      (52, 10),
        "creativity":          (58, 10),
        "analytical_thinking": (88, 5),
        "leadership":          (58, 12),
        "spatial_ability":     (84, 6),
        "memory_retention":    (70, 10),
        "empathy":             (48, 12),
        "interest_physics":    (91, 5),
        "interest_chemistry":  (74, 9),
        "interest_biology":    (38, 11),
        "interest_maths":      (93, 4),
        "interest_literature": (35, 11),
        "interest_economics":  (50, 12),
        "interest_arts":       (30, 11),
        "interest_computers":  (91, 5),
        "interest_history":    (33, 11),
        "interest_psychology": (38, 11),
        "interest_business":   (52, 11),
    },
    "Medicine": {
        "logical_reasoning":   (78, 7),
        "numerical_ability":   (70, 9),
        "verbal_ability":      (72, 8),
        "creativity":          (58, 10),
        "analytical_thinking": (82, 6),
        "leadership":          (65, 11),
        "spatial_ability":     (58, 10),
        "memory_retention":    (90, 4),
        "empathy":             (91, 4),
        "interest_physics":    (68, 9),
        "interest_chemistry":  (84, 6),
        "interest_biology":    (94, 4),
        "interest_maths":      (68, 9),
        "interest_literature": (48, 12),
        "interest_economics":  (38, 12),
        "interest_arts":       (36, 12),
        "interest_computers":  (48, 12),
        "interest_history":    (42, 12),
        "interest_psychology": (80, 6),
        "interest_business":   (35, 12),
    },
    "Commerce_CA": {
        "logical_reasoning":   (70, 8),
        "numerical_ability":   (84, 5),
        "verbal_ability":      (70, 8),
        "creativity":          (55, 8),
        "analytical_thinking": (78, 7),
        "leadership":          (72, 9),
        "spatial_ability":     (40, 8),
        "memory_retention":    (75, 8),
        "empathy":             (58, 9),
        "interest_physics":    (38, 9),
        "interest_chemistry":  (35, 9),
        "interest_biology":    (30, 9),
        "interest_maths":      (82, 6),
        "interest_literature": (52, 10),
        "interest_economics":  (91, 5),
        "interest_arts":       (38, 9),
        "interest_computers":  (50, 9),
        "interest_history":    (40, 9),
        "interest_psychology": (40, 9),
        "interest_business":   (92, 4),
    },
    "Arts_Humanities": {
        "logical_reasoning":   (52, 10),
        "numerical_ability":   (44, 10),
        "verbal_ability":      (91, 4),
        "creativity":          (90, 4),
        "analytical_thinking": (62, 10),
        "leadership":          (68, 11),
        "spatial_ability":     (65, 10),
        "memory_retention":    (72, 9),
        "empathy":             (84, 6),
        "interest_physics":    (30, 11),
        "interest_chemistry":  (28, 11),
        "interest_biology":    (36, 11),
        "interest_maths":      (38, 11),
        "interest_literature": (94, 4),
        "interest_economics":  (55, 12),
        "interest_arts":       (92, 4),
        "interest_computers":  (35, 11),
        "interest_history":    (88, 5),
        "interest_psychology": (80, 7),
        "interest_business":   (44, 12),
    },
    "Law": {
        "logical_reasoning":   (83, 6),
        "numerical_ability":   (55, 10),
        "verbal_ability":      (92, 4),
        "creativity":          (68, 9),
        "analytical_thinking": (82, 6),
        "leadership":          (83, 7),
        "spatial_ability":     (50, 11),
        "memory_retention":    (87, 5),
        "empathy":             (72, 9),
        "interest_physics":    (32, 11),
        "interest_chemistry":  (30, 11),
        "interest_biology":    (32, 11),
        "interest_maths":      (46, 12),
        "interest_literature": (87, 5),
        "interest_economics":  (72, 9),
        "interest_arts":       (55, 12),
        "interest_computers":  (42, 12),
        "interest_history":    (85, 5),
        "interest_psychology": (68, 10),
        "interest_business":   (62, 10),
    },
    "Design_Architecture": {
        "logical_reasoning":   (62, 9),
        "numerical_ability":   (60, 9),
        "verbal_ability":      (66, 9),
        "creativity":          (94, 4),
        "analytical_thinking": (66, 9),
        "leadership":          (62, 12),
        "spatial_ability":     (94, 4),
        "memory_retention":    (60, 11),
        "empathy":             (64, 10),
        "interest_physics":    (58, 11),
        "interest_chemistry":  (40, 12),
        "interest_biology":    (35, 12),
        "interest_maths":      (62, 10),
        "interest_literature": (62, 11),
        "interest_economics":  (44, 12),
        "interest_arts":       (94, 4),
        "interest_computers":  (70, 9),
        "interest_history":    (48, 12),
        "interest_psychology": (50, 12),
        "interest_business":   (46, 12),
    },
}

def generate_dataset(n_per_class=200):
    records = []
    sid = 1
    for career, archetype in ARCHETYPES.items():
        for _ in range(n_per_class):
            row = {"student_id": sid, "career_path": career}
            for feature, (mean, std) in archetype.items():
                row[feature] = likert_sample(mean, std)
            records.append(row)
            sid += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("student_data.csv", index=False)
    print(f"✅ Generated {len(df)} student records across {df['career_path'].nunique()} career paths.")
    print(df["career_path"].value_counts())
