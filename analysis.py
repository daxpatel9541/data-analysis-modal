import streamlit as st

@st.cache_data
def generate_summary(df):
    total_sales = df["Total_Sales"].sum()

    best_product = df.groupby("Product")["Total_Sales"].sum().idxmax()
    worst_product = df.groupby("Product")["Total_Sales"].sum().idxmin()

    return {
        "Total Rows": len(df),
        "Total Products": int(df["Product"].nunique()),
        "Total Sales": float(total_sales),
        "Average Sale per Transaction": float(df["Total_Sales"].mean()),
        "Best Selling Product": best_product,
        "Worst Selling Product": worst_product
    }

@st.cache_data
def top_low_products(df, n=10):
    product_sales = df.groupby("Product")["Total_Sales"].sum().sort_values(ascending=False)

    top_df = product_sales.head(n).reset_index()
    low_df = product_sales.tail(n).reset_index()

    top_df.columns = ["Product", "Total_Sales"]
    low_df.columns = ["Product", "Total_Sales"]

    return top_df, low_df

@st.cache_data
def product_sales_summary(df):
    summary = df.groupby("Product").agg(
        Total_Sales=("Total_Sales", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Avg_Sales=("Total_Sales", "mean"),
        Transaction_Count=("Total_Sales", "count")
    ).reset_index()

    total_sales_overall = summary["Total_Sales"].sum()
    summary["Contribution_Percentage"] = (summary["Total_Sales"] / total_sales_overall) * 100

    summary = summary.sort_values("Total_Sales", ascending=False)
    return summary
