from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Make sure this import is present
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the dataset
hcare = pd.read_excel("1645792390_cep1_dataset.xlsx")
hcare = hcare.dropna()

# Features and target
X = hcare.drop("target", axis=1)
y = hcare["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=7)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the logistic regression model
lr = LogisticRegression(solver='liblinear', max_iter=1000)
lr.fit(X_train_scaled, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        new_data = np.array(data).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)

        # Predict probabilities for the new data
        proba_new_data = lr.predict_proba(new_data_scaled)[:, 1] * 100
        pred_new_data = lr.predict(new_data_scaled)

        result = {
            "prediction": int(pred_new_data[0]),
            "risk_score": float(proba_new_data[0]),
            "risk_category": categorize_risk(proba_new_data[0])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

def categorize_risk(probability):
    if 0 <= probability < 30:
        return "No Risk"
    elif 30 <= probability < 50:
        return "Low Risk"
    elif 50 <= probability < 80:
        return "High Risk"
    else:
        return "Higher Risk"

if __name__ == '__main__':
    app.run(debug=True)
