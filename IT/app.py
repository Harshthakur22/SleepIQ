from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    labels = ["Poor Sleep", "Fair Sleep", "Good Sleep"]

    if request.method == "POST":
        try:
            data = [float(request.form.get(f)) for f in features]
            scaled = scaler.transform([data])
            pred = model.predict(scaled)[0]
            prediction = labels[pred]
        except:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction, features=features)

if __name__ == "__main__":
    app.run(debug=True)
