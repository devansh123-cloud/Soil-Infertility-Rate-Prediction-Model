import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\devan\Soil-Infertility-Rate-Prediction-Model\Crop_recommendation.csv")

# 2. Define Features (X) and Target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file using joblib
joblib.dump(model, "soil_crop_model.pkl")
print("Model successfully saved as 'soil_crop_model.pkl'!")