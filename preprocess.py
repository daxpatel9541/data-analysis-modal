import pandas as pd
from dateutil import parser
import streamlit as st

@st.cache_data
def preprocess_data(df, date_col, product_col, qty_col, price_col, sales_col=None):
    df = df.copy()

    selected_cols = [date_col, product_col, qty_col, price_col]
    if sales_col:
        selected_cols.append(sales_col)

    df = df[selected_cols]

    df.columns = ["Date", "Product", "Quantity", "Price"] + (["Total_Sales"] if sales_col else [])

    # Robust Date Parsing
    df["Date"] = df["Date"].apply(lambda x: parser.parse(str(x), fuzzy=True) if pd.notna(x) else pd.NaT)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric values
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if sales_col:
        df["Total_Sales"] = pd.to_numeric(df["Total_Sales"], errors="coerce")

    # Remove invalid rows
    df.dropna(inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    if sales_col:
        df = df[df["Total_Sales"] > 0]

    df.drop_duplicates(inplace=True)

    # Compute Total_Sales if not given
    if not sales_col:
        df["Total_Sales"] = df["Quantity"] * df["Price"]

    return df
