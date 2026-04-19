import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

print("Generating NYC housing dataset...")

np.random.seed(42)
n = 3000

boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
neighborhoods = {
    'Manhattan':    ['Midtown', 'Upper East Side', 'Harlem', 'Financial District', 'Chelsea', 'SoHo'],
    'Brooklyn':     ['Williamsburg', 'Park Slope', 'Flatbush', 'Bushwick', 'DUMBO', 'Bay Ridge'],
    'Queens':       ['Astoria', 'Flushing', 'Jamaica', 'Forest Hills', 'Long Island City', 'Jackson Heights'],
    'Bronx':        ['Fordham', 'Riverdale', 'South Bronx', 'Pelham Bay', 'Mott Haven', 'Norwood'],
    'Staten Island':['St. George', 'Stapleton', 'Great Kills', 'Tottenville', 'New Dorp', 'Annadale'],
}
base_prices = {
    'Manhattan': 1_200_000,
    'Brooklyn':  750_000,
    'Queens':    600_000,
    'Bronx':     400_000,
    'Staten Island': 480_000,
}

borough_choices = np.random.choice(boroughs, n).tolist()
neighborhood_choices = [np.random.choice(neighborhoods[b]) for b in borough_choices]
sqft      = np.random.randint(400, 3500, n)
bedrooms  = np.random.randint(0, 6, n)
bathrooms = np.random.randint(1, 5, n)
floor     = np.random.randint(1, 50, n)
age       = np.random.randint(0, 100, n)
has_garage    = np.random.randint(0, 2, n)
has_elevator  = np.random.randint(0, 2, n)
has_doorman   = np.random.randint(0, 2, n)
subway_dist   = np.round(np.random.uniform(0.1, 2.5, n), 2)

price = np.array([base_prices[b] for b in borough_choices], dtype=float)

price += sqft * np.random.uniform(200, 600, n)
price += bedrooms * np.random.uniform(10000, 40000, n)
price += bathrooms * np.random.uniform(8000, 25000, n)
price += floor * np.random.uniform(1000, 5000, n)
price -= age * np.random.uniform(500, 3000, n)
price += has_garage * np.random.uniform(10000, 30000, n)
price += has_elevator * np.random.uniform(5000, 15000, n)
price += has_doorman * np.random.uniform(8000, 20000, n)
price -= subway_dist * np.random.uniform(10000, 30000, n)
price += np.random.normal(0, 30000, n)
price = np.clip(price, 80000, 8_000_000)

df = pd.DataFrame({
    'borough':       borough_choices,
    'neighborhood':  neighborhood_choices,
    'sqft':          sqft,
    'bedrooms':      bedrooms,
    'bathrooms':     bathrooms,
    'floor':         floor,
    'age':           age,
    'has_garage':    has_garage,
    'has_elevator':  has_elevator,
    'has_doorman':   has_doorman,
    'subway_dist':   subway_dist,
    'price':         price.astype(int),
})

os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)
df.to_csv(os.path.join(os.path.dirname(__file__), 'nyc_housing.csv'), index=False)
print(f"Dataset created: {len(df)} rows")

le_borough = LabelEncoder()
le_neighborhood = LabelEncoder()
df['borough_enc']      = le_borough.fit_transform(df['borough'])
df['neighborhood_enc'] = le_neighborhood.fit_transform(df['neighborhood'])

features = ['borough_enc','neighborhood_enc','sqft','bedrooms','bathrooms',
            'floor','age','has_garage','has_elevator','has_doorman','subway_dist']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Training Gradient Boosting model...")
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : ${rmse:,.0f}")

out = os.path.dirname(__file__)
pickle.dump(model,         open(os.path.join(out, 'model.pkl'),         'wb'))
pickle.dump(scaler,        open(os.path.join(out, 'scaler.pkl'),        'wb'))
pickle.dump(le_borough,    open(os.path.join(out, 'le_borough.pkl'),    'wb'))
pickle.dump(le_neighborhood, open(os.path.join(out, 'le_neighborhood.pkl'), 'wb'))
print("Model and encoders saved!")
print("\nTraining complete. You can now run: uvicorn app:app --reload")