import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, os

df = pd.read_csv("pilgrimage_data.csv")

day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weather_order = ["Stormy","Rainy","Foggy","Cold","Very Hot","Hot","Warm","Cloudy","Cool","Pleasant"]
festival_order = ["none","minor","major"]

df["day_num"] = df["day"].apply(lambda x: day_order.index(x))
df["weather_num"] = df["weather"].apply(lambda x: weather_order.index(x) if x in weather_order else 5)
df["festival_num"] = df["festival_type"].apply(lambda x: festival_order.index(x))

features = ["day_num","month","is_weekend","festival_num","weather_num"]
X = df[features]
y = df["footfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.0f} pilgrims | R2 Score: {r2:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

meta = {
    "features": features,
    "day_order": day_order,
    "weather_order": weather_order,
    "festival_order": festival_order,
}
with open("model_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Model saved: model.pkl")
print(f"Feature importances: {dict(zip(features, model.feature_importances_.round(3)))}")
