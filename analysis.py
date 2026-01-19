def generate_summary(df):
    return {
        "Total Rows": len(df),
        "Total Products": int(df["Product"].nunique()),
        "Total Sales": float(df["Total_Sales"].sum()),
        "Average Sale per Transaction": float(df["Total_Sales"].mean())
    }

def top_low_products(df, n=10):
    product_sales = df.groupby("Product")["Total_Sales"].sum().sort_values(ascending=False)

    top_df = product_sales.head(n).reset_index()
    low_df = product_sales.tail(n).reset_index()

    top_df.columns = ["Product", "Total_Sales"]
    low_df.columns = ["Product", "Total_Sales"]

    return top_df, low_df
