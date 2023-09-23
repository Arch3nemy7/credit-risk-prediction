import streamlit as st
import pandas as pd
import joblib

st.header('FTDS Model Deployment')
st.write("""
Created by FTDS Curriculum Team
Use the sidebar to select input features.
""")

@st.cache_data
def fetch_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Freezer9/Dataset-ISE/main/credit_risk.csv')
    return df

def label_encode(data):
    return data.replace({'Y': 1, 'N': 0})

df = fetch_data()
st.write(df)

st.sidebar.header('User Input Features')


def user_input():
    age = st.sidebar.number_input('Age', 0, value=0)
    income = st.sidebar.number_input('Income', 0, value=0)
    home = st.sidebar.selectbox('Home', df['Home'].unique())
    emp_length = st.sidebar.number_input('Emp_length', 0, value=0)
    intent = st.sidebar.selectbox('Intent', df['Intent'].unique())
    amount = st.sidebar.number_input('Amount', 0, value=0)
    rate = st.sidebar.number_input('Rate', 0, value=0)
    percent_income = st.sidebar.number_input('Percent_income', 0.0, value=0.0)
    default = st.sidebar.selectbox('Default', df['Default'].unique())
    cred_length = st.sidebar.number_input('Cred_length', 0, value=0)

    data = {
        'Age': age,
        'Income': income,
        'Home': home,
        'Emp_length': emp_length,
        'Intent': intent,
        'Amount': amount,
        'Rate': rate,
        'Percent_income': percent_income,
        'Default': default,
        'Cred_length': cred_length
    }
    features = pd.DataFrame(data, index=[0])    
    return features


input = user_input()

st.subheader('User Input')
st.write(input)

load_model = joblib.load("credit_risk_model.pkl")

if st.button('Predict'):
    prediction = load_model.predict(input)

    if prediction == 1:
        prediction = 'Trusted'
    else:
        prediction = 'Not Trusted'
    
    st.write('Based on user input, the placement model predicted: ')
    st.write(prediction)
else:
    pass