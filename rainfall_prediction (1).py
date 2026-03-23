# Rainfall Prediction Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/weatherAUS.csv"
data = pd.read_csv(url)

# Save dataset
data.to_csv("rainfall_data.csv", index=False)

# Preprocess
data = data.dropna()
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Features
features = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'Pressure3pm']
X = data[features]
y = data['RainTomorrow']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
