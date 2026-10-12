from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/columns.pkl", "rb") as f:
    expectedColumns = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return "Disease Risk Predictor API is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        inputDf = pd.DataFrame([data])

        for col in expectedColumns:
            if col not in inputDf.columns:
                inputDf[col] = 0

        inputDf = inputDf[expectedColumns]

        prediction = model.predict(inputDf)[0]
        probability = model.predict_proba(inputDf)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "riskProbability": round(float(probability), 3),
            "result": "High risk of diabetes" if prediction == 1 else "Low risk of diabetes"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)