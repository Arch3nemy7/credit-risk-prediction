import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def fetch_data():
    url = "https://raw.githubusercontent.com/Freezer9/My-Dataset/refs/heads/main/Dataset-ISE/credit_risk.csv"
    return pd.read_csv(url)

def label_encode(data):
    return data.replace({"Y": 1, "N": 0})

def user_input():
    age = st.sidebar.number_input("Age", min_value=0, value=20)
    income = st.sidebar.number_input("Income", min_value=0, value=4000, step=1000)
    home = st.sidebar.selectbox("Home", df["Home"].unique())
    emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0, value=5)
    intent = st.sidebar.selectbox("Loan Intent", df["Intent"].unique())
    amount = st.sidebar.number_input("Loan Amount", min_value=0, value=10000, step=1000)
    rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=10.0, step=1.0)
    percent_income = st.sidebar.number_input("Loan as % of Income", min_value=0.0, value=0.20, step=0.10)
    default = st.sidebar.selectbox("Default History", df["Default"].unique())
    cred_length = st.sidebar.number_input("Credit Length (years)", min_value=0, value=5)

    data = {
        "Age": age,
        "Income": income,
        "Home": home,
        "Emp_length": emp_length,
        "Intent": intent,
        "Amount": amount,
        "Rate": rate,
        "Percent_income": percent_income,
        "Default": default,
        "Cred_length": cred_length,
    }

    return pd.DataFrame(data, index=[0])

def preprocess_input(input_df):
    input_df["Home"] = label_encode(input_df["Home"])
    input_df["Intent"] = label_encode(input_df["Intent"])
    input_df["Default"] = label_encode(input_df["Default"])
    return input_df

st.title("Credit Risk Model Deployment")
st.write("Created by Jawir Team")
st.write("Use the sidebar to select input features.")

df = fetch_data()
st.dataframe(df, hide_index=True)

st.sidebar.header("User Input Features")

input_features = user_input()
st.subheader("User Input")
st.dataframe(input_features)

processed_input = preprocess_input(input_features)

model_path = "credit_risk_model.pkl"
load_model = joblib.load(model_path)

if st.button("Predict"):
    prediction = load_model.predict(processed_input)
    
    st.write("Based on user input, the model predicted:")
    if prediction[0] == 1:
        st.success("Trusted")
    else:
        st.error("Not Trusted")
else:
    st.info("Click 'Predict' to see the results.")