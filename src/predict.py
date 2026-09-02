import pickle

# Load trained model
with open("models/diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_diabetes():

    pregnancies = int(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

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
        print("\nDiabetes predicted")
    else:
        print("\nNo diabetes predicted")

    print(f"Diabetes probability: {probability * 100:.2f}%")

if __name__ == "__main__":
    predict_diabetes()