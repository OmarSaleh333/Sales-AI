import streamlit as st
import pickle
import pandas as pd

st.title("This is our Website.")

st.subheader("Ecommerce 🧺")

with open("decisiontree.pkl", "rb") as f:
    model = pickle.load(f)


df = pd.read_csv('ecommerce_sales_data.csv')

# st.table(df.head())

st.dataframe(df.head())

df.columns = df.columns.str.replace(' ' , '-')
df['Order-Date'] = df['Order-Date'].astype('date64[pyarrow]')
df['Year'] = df['Order-Date'].dt.year
df['Month'] = df['Order-Date'].dt.month
df['Day'] = df['Order-Date'].dt.day
df.drop(columns='Order-Date' , inplace=True)

st.divider()

st.sidebar.subheader("Try Our Prediction Model AI")

product_name = st.sidebar.selectbox('Product Name:',df['Product-Name'].unique())

category = df[df['Product-Name']==product_name]['Category'].unique()[0]

region = st.sidebar.selectbox('Region:',df['Region'].unique())

quantity = st.sidebar.number_input('Quantity' , min_value = 1)

sales = st.sidebar.number_input("Sales" , min_value= df['Sales'].min() , help="Enter a sales number")

date = st.sidebar.date_input("Date")


user_data = pd.DataFrame([{
    'Product-Name': product_name,
    'Category': category,
    'Region': region,
    'Quantity': quantity,
    'Sales': sales,
    'Year': date.year,
    'Month': date.month,
    'Day': date.day}])

prediction = model.predict(user_data)

if st.sidebar.button('Predict'):
    st.sidebar.success(f'The Profit Predict = {prediction[0].round(2)}')
    st.balloons()