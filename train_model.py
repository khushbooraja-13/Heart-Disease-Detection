# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data
df = pd.read_csv("heart.csv")

# 2. Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# 5. Save model and scaler ➡️ THIS IS WHERE YOU WRITE IT
joblib.dump(model, 'model/heart_disease_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# 6. Evaluate (optional)
print("Model saved successfully!")