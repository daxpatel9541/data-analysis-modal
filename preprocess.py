import pandas as pd

def preprocess_data(df, date_col, product_col, qty_col, price_col):
    df = df.copy()

    df = df[[date_col, product_col, qty_col, price_col]]
    df.columns = ["Date", "Product", "Quantity", "Price"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df.dropna(inplace=True)

    df["Total_Sales"] = df["Quantity"] * df["Price"]
    return df
