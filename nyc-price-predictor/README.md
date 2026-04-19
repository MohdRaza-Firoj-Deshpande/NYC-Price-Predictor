# 🗽 NYC House Price Predictor

Predict New York City property prices using a Gradient Boosting ML model.
Built with FastAPI + scikit-learn + vanilla HTML/CSS/JS.

---

## 🚀 Run in VS Code — Step by Step

### 1. Open the folder in VS Code
```
File → Open Folder → select "nyc-price-predictor"
```

### 2. Open the integrated terminal
```
Ctrl + ` (backtick)
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the ML model (do this once)
```bash
python model/train.py
```
This generates `model.pkl`, `scaler.pkl`, and encoder files inside the `model/` folder.

### 6. Start the FastAPI server
```bash
uvicorn app:app --reload
```

### 7. Open in browser
```
http://127.0.0.1:8000
```

---

## 📁 Project Structure

```
nyc-price-predictor/
├── app.py                  ← FastAPI backend (main server)
├── requirements.txt        ← Python dependencies
├── predictions.db          ← SQLite DB (auto-created)
│
├── model/
│   ├── train.py            ← Train the ML model
│   ├── model.pkl           ← Trained model (after training)
│   ├── scaler.pkl          ← Feature scaler
│   ├── le_borough.pkl      ← Borough label encoder
│   ├── le_neighborhood.pkl ← Neighborhood label encoder
│   └── nyc_housing.csv     ← Synthetic NYC dataset
│
├── templates/
│   └── index.html          ← Main HTML page
│
└── static/
    ├── css/style.css       ← Styling
    └── js/main.js          ← Frontend logic
```

---

## 🧠 Model Details

| Item | Detail |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Dataset | Synthetic NYC housing (3,000 rows) |
| Features | Borough, Neighborhood, Sqft, Beds, Baths, Floor, Age, Amenities, Subway distance |
| Typical R² | ~0.90+ |

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main web UI |
| POST | `/api/predict` | Predict price (JSON body) |
| GET | `/api/history` | Last 8 predictions |
| GET | `/api/neighborhoods/{borough}` | Get neighborhoods for a borough |

---

## 🌐 Deploy Free

| Platform | Command |
|---|---|
| Render.com | Connect GitHub repo → set start command to `uvicorn app:app --host 0.0.0.0 --port 10000` |
| Railway.app | `railway up` |
