import sqlite3, json, os, pickle
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

BASE  = os.path.dirname(__file__)
MODEL = os.path.join(BASE, "model")

app = FastAPI(title="NYC House Price Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))

def load():
    needed = ["model.pkl","scaler.pkl","le_borough.pkl","le_neighborhood.pkl"]
    for f in needed:
        if not os.path.exists(os.path.join(MODEL, f)):
            return None, None, None, None
    m  = pickle.load(open(os.path.join(MODEL,"model.pkl"),          "rb"))
    s  = pickle.load(open(os.path.join(MODEL,"scaler.pkl"),         "rb"))
    lb = pickle.load(open(os.path.join(MODEL,"le_borough.pkl"),     "rb"))
    ln = pickle.load(open(os.path.join(MODEL,"le_neighborhood.pkl"),"rb"))
    return m, s, lb, ln

model, scaler, le_borough, le_neighborhood = load()

NEIGHBORHOODS = {
    "Manhattan":     ["Midtown","Upper East Side","Harlem","Financial District","Chelsea","SoHo"],
    "Brooklyn":      ["Williamsburg","Park Slope","Flatbush","Bushwick","DUMBO","Bay Ridge"],
    "Queens":        ["Astoria","Flushing","Jamaica","Forest Hills","Long Island City","Jackson Heights"],
    "Bronx":         ["Fordham","Riverdale","South Bronx","Pelham Bay","Mott Haven","Norwood"],
    "Staten Island": ["St. George","Stapleton","Great Kills","Tottenville","New Dorp","Annadale"],
}

DB = os.path.join(BASE, "predictions.db")

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_data TEXT, predicted_price INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def save_pred(data, price):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO predictions (input_data,predicted_price) VALUES (?,?)",
                 (json.dumps(data), int(price)))
    conn.commit(); conn.close()

def get_history():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id,input_data,predicted_price,timestamp FROM predictions ORDER BY id DESC LIMIT 8"
    ).fetchall()
    conn.close()
    return [{"id":r[0],"input":json.loads(r[1]),"price":r[2],"ts":r[3]} for r in rows]

init_db()

class HouseIn(BaseModel):
    borough: str
    neighborhood: str
    sqft: int
    bedrooms: int
    bathrooms: int
    floor: int
    age: int
    has_garage: int
    has_elevator: int
    has_doorman: int
    subway_dist: float

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"neighborhoods": NEIGHBORHOODS, "model_ready": model is not None}
    )

@app.get("/api/neighborhoods/{borough}")
def neighborhoods(borough: str):
    return {"neighborhoods": NEIGHBORHOODS.get(borough, [])}

@app.post("/api/predict")
def predict(h: HouseIn):
    if model is None:
        return {"error": "Model not trained yet. Run: python model/train.py"}
    try:
        b_enc = le_borough.transform([h.borough])[0]
        n_enc = le_neighborhood.transform([h.neighborhood])[0]
    except ValueError:
        return {"error": "Unknown borough or neighborhood"}
    X = np.array([[b_enc, n_enc, h.sqft, h.bedrooms, h.bathrooms,
                   h.floor, h.age, h.has_garage, h.has_elevator,
                   h.has_doorman, h.subway_dist]])
    X_s = scaler.transform(X)
    pred = int(model.predict(X_s)[0])
    low  = int(pred * 0.90)
    high = int(pred * 1.10)
    per_sqft = int(pred / h.sqft) if h.sqft > 0 else 0
    save_pred(h.model_dump(), pred)
    return {"predicted_price": pred, "low": low, "high": high,
            "per_sqft": per_sqft, "borough": h.borough}

@app.get("/api/history")
def history():
    return {"history": get_history()}