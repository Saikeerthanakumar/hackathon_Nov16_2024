
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib
from fastapi import HTTPException


app =  FastAPI()

class Input(BaseModel):
    department: object
    region: object
    education: object
    gender: object
    recruitment_channel: object
    no_of_trainings: int
    age: int
    previous_year_rating: float
    length_of_service: int
    KPIs_met_gt_80: int
    awards_won: int
    avg_training_score: int
    
    

class Output(BaseModel):
    is_promoted: int

@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([[data.department,data.region,data.education,data.gender,data.recruitment_channel,data.no_of_trainings,data.age,data.previous_year_rating,data.length_of_service,data.KPIs_met_gt_80,data.awards_won,data.avg_training_score]])
    X_input.columns = ['department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met_gt_80','awards_won','avg_training_score']
    print(X_input.columns)
    
#load model
    model = joblib.load('jobpromo_pipeline_xgbmodel.pkl')
   ## model = joblib.load('promo_pipeline_model.pkl')
    
    #predict
    prediction = model.predict(X_input)
        
    #output
    return Output(is_promoted = prediction)
    

  