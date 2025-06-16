import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 

df = pd.read_csv("insurance.csv")  

# Feature engineering
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# Defining features
num_features = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
cat_features = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants',
                'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']

X = df[num_features + cat_features]
y = df['PremiumPrice']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Training model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Saving model and scaler locally
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('insurance_best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Scaler and model saved successfully")
