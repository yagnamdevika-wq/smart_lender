import os
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Train chesina files ni load cheskuntunnam
model = pickle.load(open('Flask/rdf.pkl', 'rb'))
scaler = pickle.load(open('Flask/scale1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form nunchi anni inputs ni gather cheyadam
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = int(request.form['applicant_income'])
        coapplicant_income = int(request.form['coapplicant_income'])
        loan_amount = int(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        credit_history = int(request.form['credit_history'])
        property_area = int(request.form['property_area'])
        
        # Array la marcharam
        features = np.array([[gender, married, dependents, education, self_employed,
                              applicant_income, coapplicant_income, loan_amount, 
                              loan_term, credit_history, property_area]])
        
        # Scaling apply cheyadam
        scaled_features = scaler.transform(features)
        
        # Model Prediction
        prediction = model.predict(scaled_features)
        
        if int(prediction[0]) == 0:
            result_text = "Loan will Not be Approved"
        else:
            result_text = "Loan will be Approved"
            
        return render_template('output.html', result=result_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)