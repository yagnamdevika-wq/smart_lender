import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Dataset ni load cheyadam
data = pd.read_csv(r'C:\Users\kamma\OneDrive\Desktop\smartlender\loan_prediction.csv')

# 2. Null values (Missing values) ni handle cheyadam
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])

# Dependents column lo '3+' unte '+' character ni remove cheyadam
data['Dependents'] = data['Dependents'].str.replace('+', '', regex=False)
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])

data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# 3. Text (Categorical) data ni Numbers loki map cheyadam
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Datatypes ni integer ki convert cheyadam
columns_to_cast = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'ApplicantIncome', 
                   'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in columns_to_cast:
    data[col] = data[col].astype('int64')

# 4. Features (X) and Target (y) ga split cheyadam
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# 5. Feature Scaling (Data ni oke range loki techukovadam)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scaler object ni save cheyadam
with open('Flask/scale1.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 6. Train-Test Split & Random Forest Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model object ni save cheyadam
with open('Flask/rdf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Super! Model and Scaler successfully training ayyi Flask folder lo save ayyayi!")