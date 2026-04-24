from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from predict import predict_top3, FEATURES

app = FastAPI(title="Career Guidance API")

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


class StudentScores(BaseModel):
    logical_reasoning: float = Field(..., ge=0, le=100)
    numerical_ability: float = Field(..., ge=0, le=100)
    verbal_ability: float = Field(..., ge=0, le=100)
    creativity: float = Field(..., ge=0, le=100)
    analytical_thinking: float = Field(..., ge=0, le=100)
    leadership: float = Field(..., ge=0, le=100)
    spatial_ability: float = Field(..., ge=0, le=100)
    memory_retention: float = Field(..., ge=0, le=100)
    empathy: float = Field(..., ge=0, le=100)
    interest_physics: float = Field(..., ge=0, le=100)
    interest_chemistry: float = Field(..., ge=0, le=100)
    interest_biology: float = Field(..., ge=0, le=100)
    interest_maths: float = Field(..., ge=0, le=100)
    interest_literature: float = Field(..., ge=0, le=100)
    interest_economics: float = Field(..., ge=0, le=100)
    interest_arts: float = Field(..., ge=0, le=100)
    interest_computers: float = Field(..., ge=0, le=100)
    interest_history: float = Field(..., ge=0, le=100)
    interest_psychology: float = Field(..., ge=0, le=100)
    interest_business: float = Field(..., ge=0, le=100)


@app.get("/")
def root():
    return FileResponse(_STATIC / "index.html")


@app.post("/predict")
def predict(scores: StudentScores):
    try:
        results = predict_top3(scores.model_dump())
        return {
            "recommendations": [
                {"career": career.replace("_", " "), "confidence": round(float(prob) * 100, 1)}
                for career, prob in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
