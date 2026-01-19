import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from utils.mapper import auto_detect_columns
from utils.preprocess import preprocess_data
from utils.analysis import generate_summary, top_low_products
from utils.forecasting import train_model, predict_future_sales

st.set_page_config(page_title="Universal Company Analyzer", layout="wide")
st.title("📊 Universal Company Data Analyzer (Works with ANY Dataset)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🧠 Auto Column Detection")
    detected = auto_detect_columns(df.columns)

    st.write("Detected Mapping:", detected)

    st.subheader("🛠 Column Mapping (Select Correct Columns)")
    cols = list(df.columns)

    date_col = st.selectbox("Select Date Column", cols, index=cols.index(detected["date"]) if detected["date"] in cols else 0)
    product_col = st.selectbox("Select Product Column", cols, index=cols.index(detected["product"]) if detected["product"] in cols else 0)
    qty_col = st.selectbox("Select Quantity Column", cols, index=cols.index(detected["quantity"]) if detected["quantity"] in cols else 0)
    price_col = st.selectbox("Select Price/Sales Column", cols, index=cols.index(detected["price"]) if detected["price"] in cols else 0)

    st.subheader("✅ Process Dataset")
    df_clean = preprocess_data(df, date_col, product_col, qty_col, price_col)

    st.success("Dataset processed successfully!")
    st.dataframe(df_clean.head())

    # Summary
    st.subheader("📈 Company Summary Report")
    summary = generate_summary(df_clean)
    st.json(summary)

    # Top / Low products
    st.subheader("🔥 Top Selling vs Low Selling Products")
    top_df, low_df = top_low_products(df_clean)

    c1, c2 = st.columns(2)
    with c1:
        st.write("✅ Top Products")
        st.dataframe(top_df)

    with c2:
        st.write("❌ Low Products")
        st.dataframe(low_df)

    # Trend chart
    st.subheader("📊 Sales Trend")
    daily_sales = df_clean.groupby("Date")["Total_Sales"].sum()

    fig, ax = plt.subplots()
    daily_sales.plot(ax=ax)
    ax.set_title("Sales Over Time")
    ax.set_ylabel("Total Sales")
    st.pyplot(fig)

    # Train Model
    st.subheader("🤖 Train Prediction Model")
    if st.button("Train Model"):
        model = train_model(df_clean)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/sales_model.pkl")
        st.success("Model trained & saved!")

    # Predict
    if os.path.exists("models/sales_model.pkl"):
        model = joblib.load("models/sales_model.pkl")

        future_days = st.number_input("Predict next N days", min_value=1, max_value=60, value=7)

        if st.button("Predict Future Sales"):
            future_df = predict_future_sales(df_clean, model, future_days)
            st.dataframe(future_df)
