import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="karthick2613/superkart_mlops", filename="best_product_sales_xgb_regressor_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Sales Prediction
st.title("Sales Forecasting App")
st.write("""
This application forecasts the product sales based on the product related inputs. Please enter the product and store related inputs below
""")

# User input
Product_Weight = st.number_input("Product_Weight", min_value=1, step=0.01, value=1)
Product_Allocated_Area = st.number_input("Product_Allocated_Area", min_value=0, step=0.001, value=0.001)
Product_MRP = st.number_input("Product_MRP", min_value=1, step=0.5, value=1)
Store_Establishment_Year = st.number_input("Store_Establishment_Year", min_value=1900, step=1, value=2000)
Product_Type = st.selectbox("Product_Type", ["Fruits and Vegetables", "Snack Foods ", "Frozen Foods"])
Product_Sugar_Content = st.selectbox("Product_Sugar_Content", ["Low Sugar", "Regular", "No Sugar", "reg"])
Store_Size = st.selectbox("Store_Size", ["Medium", "High", "Small"])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store_Type", ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Product_Type': Product_Type,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])


# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted sales for the product is ${np.exp(prediction)[0]:.2f}.")
