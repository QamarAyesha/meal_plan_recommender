from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import os

app = FastAPI()

# Load model and assets
model_path = os.path.join(os.path.dirname(_file_), "model", "meal_recommender_rf.pkl")
encoders_path = os.path.join(os.path.dirname(_file_), "model", "label_encoders.pkl")
meal_ideas_path = os.path.join(os.path.dirname(_file_), "model", "meal_ideas.json")

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)
with open(meal_ideas_path) as f:
    meal_ideas = json.load(f)

# Request model
class UserInput(BaseModel):
    age: int
    region: str
    breastfeeding_stage: str
    health_condition: str

@app.post("/predict")
async def predict(user_input: UserInput):
    try:
        # Encode inputs
        region_enc = encoders['le_region'].transform([user_input.region])[0]
        stage_enc = encoders['le_stage'].transform([user_input.breastfeeding_stage])[0]
        health_enc = encoders['le_health'].transform([user_input.health_condition])[0]

        # Predict
        prediction = model.predict([[user_input.age, region_enc, stage_enc, health_enc]])[0]
        plan = encoders['le_plan'].inverse_transform([prediction])[0]

        return {
            "plan": plan,
            "meals": meal_ideas.get(plan, ["Generic balanced meal"]),
            "tips": [
                "Stay hydrated",
                "Consult a nutritionist for personalized advice"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "alive"}
