from flask import Flask, request, render_template, redirect
import joblib
import pandas as pd

app = Flask(__name__)

model_fit = joblib.load(open("./models/heart_disease.pkl", "rb"))
print("Model Loaded")
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
print("Scaler Model Loaded")

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        if request.form.get("predict")=="predict":
            return redirect("/predict")
        if request.form.get("dashboard")=="dashboard":
            return redirect("/dashboard")        
    return render_template("home.html")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template("dashboard.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        age =   int(request.form.get('age'))
        sex =   int(request.form.get('sex'))
        cp =   int(request.form.get('cp'))
        trestbps =   int(request.form.get('trestbps'))
        chol =   int(request.form.get('chol'))
        fbs =   int(request.form.get('fbs'))
        restecg =   int(request.form.get('restecg'))
        mhr =   int(request.form.get('mhr'))
        exang =   int(request.form.get('exang'))
        stdep =   float(request.form.get('stdep'))
        slope =   int(request.form.get('slope'))
        ca =   int(request.form.get('vessels'))
        thal =   int(request.form.get('thal'))
        
        new_input = {
            'Age': age,
            'Sex': sex,
            'Chest pain type': cp,
            'BP': trestbps,
            'Cholesterol': chol,
            'FBS over 120': fbs,
            'EKG results': restecg,
            'Max HR': mhr,
            'Exercise angina': exang,
            'ST depression': stdep,
            'Slope of ST': slope,
            'Number of vessels fluro': ca,
            'Thallium': thal
        }
        print(new_input)

        
        def predict_input(input):
            input_df = pd.DataFrame([input])
            X_input = scaler.fit_transform(input_df)
            pred = model_fit.predict(X_input)[0]
            return pred
        
        prediction = predict_input(new_input)
        
        if prediction==1:
            output = "Sorry, You are at high risk of having a heart disease. Please consult a doctor as soon as possible"
        elif prediction==0:
            output = "Hi! You are at low risk of having a heart disease. If you are still not convinced please consult a doctor"
        else:
            output = "Hey, there was some error processing your details. Please try again later."

        return render_template('result.html',prediction_text="{}".format(output))
                
    return render_template("predict.html")

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
