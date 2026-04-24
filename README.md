# Career Guidance System — ML Backend

A machine learning system that recommends top 3 career paths for Class 10 & 12 students based on their aptitude and interests.

## Career Paths Covered
- Engineering
- Medicine
- Commerce / CA
- Arts & Humanities
- Law
- Design & Architecture

## Project Structure

```
career_guidance/
├── generate_data.py     # Generates synthetic student dataset
├── train_model.py       # Trains Random Forest model
├── predict.py           # Predict top 3 careers for a student
├── student_data.csv     # Generated after running generate_data.py
├── career_model.pkl     # Saved model (after training)
├── label_encoder.pkl    # Saved label encoder (after training)
└── README.md
```

## Features Used in Assessment

| Feature              | Description                        |
|----------------------|------------------------------------|
| logical_reasoning    | Aptitude for logic problems        |
| numerical_ability    | Maths and number skills            |
| verbal_ability       | Language and communication skills  |
| creativity           | Creative thinking ability          |
| analytical_thinking  | Problem analysis skills            |
| leadership           | Leadership and teamwork traits     |
| interest_physics     | Interest score in Physics          |
| interest_chemistry   | Interest score in Chemistry        |
| interest_biology     | Interest score in Biology          |
| interest_maths       | Interest score in Maths            |
| interest_literature  | Interest score in Literature       |
| interest_economics   | Interest score in Economics        |
| interest_arts        | Interest score in Arts             |

All scores are out of 100.

## How to Run

### Step 1 — Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### Step 2 — Generate synthetic data
```bash
python generate_data.py
```

### Step 3 — Train the model
```bash
python train_model.py
```

### Step 4 — Predict career for a student
```bash
python predict.py
```

## Usage in Your Web App

Import `predict_top3()` from `predict.py` and pass a dictionary of student scores:

```python
from predict import predict_top3

student = {
    "logical_reasoning": 85,
    "numerical_ability": 80,
    "verbal_ability": 60,
    "creativity": 55,
    "analytical_thinking": 82,
    "leadership": 65,
    "interest_physics": 88,
    "interest_chemistry": 72,
    "interest_biology": 40,
    "interest_maths": 90,
    "interest_literature": 38,
    "interest_economics": 50,
    "interest_arts": 32,
}

top3 = predict_top3(student)
# Returns: [("Engineering", 0.87), ("Medicine", 0.07), ("Commerce_CA", 0.04)]
```

## Next Steps
- Integrate this backend with a Flask/FastAPI web server
- Build the frontend assessment form
- Replace synthetic data with real student data over time
