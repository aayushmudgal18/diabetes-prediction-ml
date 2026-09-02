import sklearn
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


# Load the trained model
with open("models/diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    pregnancies = int(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    diabetes_pedigree = float(request.form["diabetes_pedigree"])
    age = int(request.form["age"])

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        result = "Diabetes"
    else:
        result = "No Diabetes"

    probability = round(probability * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)

