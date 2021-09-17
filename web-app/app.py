import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("../model/heart_failure_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]    
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    if prediction:
        output="You are in high risk of Heart Failure !! Consider consulting a cardiologist immediately" 
    else:
        output="You are in low risk of Heart Failure Make sure to be fit with healthy diet and exercise"    

    return render_template("index.html", prediction_text=output)
    


if __name__ == "__main__":
    app.run(debug=True)
