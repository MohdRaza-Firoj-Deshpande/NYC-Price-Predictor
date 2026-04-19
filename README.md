# NYC-Price-Predictor
A full-stack ML app:  Train a model, serve it via API, save predictions to a database, all in one repo.Predicts NYC property prices using Gradient Boosting, FastAPI, and SQLite

🗽 NYC House Price Predictor
A machine learning web app that predicts property prices across New York City boroughs. Enter details like neighborhood, size, floor, and amenities — get an instant price estimate.

<img width="1362" height="562" alt="Capture" src="https://github.com/user-attachments/assets/a41905e5-1162-4586-ba7f-6c02fec008b1" />


Tech used

scikit-learn — Gradient Boosting model  

FastAPI — backend API  

SQLite — saves prediction history  

HTML / CSS / JS — frontend, no frameworks

Run locally
pip install -r requirements.txt  

python model/train.py  

uvicorn app:app --reload  

Open http://127.0.0.1:8000

How it works
Trained on 3,000 synthetic NYC property records. Categorical features like borough and neighborhood are label encoded, all features are scaled with StandardScaler. The Gradient Boosting model achieves an R² of ~0.71 on the test set. Every prediction is saved to SQLite and shown in the history panel.
