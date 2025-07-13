import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('Data_Prediction.csv')  
X = df.drop('CabinTemperature',axis='columns')  
y = df["CabinTemperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.2, random_state=42), 
}

best_model = None
best_score = float('-inf')

print("Model Evaluation:\n------------------")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MAE = {mae:.2f}, R² = {r2:.3f}")
    
    if r2 > best_score:
        best_model = model
        best_model_name = name
        best_score = r2
        
joblib.dump(best_model, "ThermalComfort_prediction_model.pkl")
print(f"\n✅ Best model: {best_model_name} saved as 'ThermalComfort_prediction_model.pkl'")