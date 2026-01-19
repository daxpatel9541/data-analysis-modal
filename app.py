import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import chardet
from io import BytesIO

from mapper import auto_detect_columns
from preprocess import preprocess_data
from analysis import generate_summary, top_low_products, product_sales_summary
from forecasting import train_product_forecast_model, predict_product_future_sales, get_top_future_products
from report_pdf import generate_pdf_report

st.set_page_config(page_title="Universal Company Analyzer", layout="wide")
st.title("📊 Universal Company Data Analyzer (Works with ANY Dataset)")

# ---------------- SAFE CSV READER ----------------
@st.cache_data
def read_csv_safely(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    detected = chardet.detect(file_bytes)
    encoding = detected.get("encoding") or "utf-8"

    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        return df, encoding
    except Exception:
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding=enc)
                return df, enc
            except Exception:
                continue

    raise ValueError("Could not read this CSV file. Please re-save as UTF-8 or try another dataset.")

# Sidebar navigation
st.sidebar.header("📌 Navigation")
menu = st.sidebar.radio(
    "Go To",
    ["Upload & Process", "Sales Analytics", "Product Insights", "Forecasting (Product Wise)", "Downloads"]
)

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df, used_encoding = read_csv_safely(uploaded_file)
        st.success(f"✅ File loaded successfully (Encoding: {used_encoding})")
    except Exception as e:
        st.error(f"❌ CSV Read Error: {e}")
        st.stop()

    # ---------------- Upload & Process ----------------
    if menu == "Upload & Process":
        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("🧠 Auto Column Detection")
        detected = auto_detect_columns(df.columns)
        st.write("Detected Mapping:", detected)

        st.subheader("🛠 Column Mapping (Select Correct Columns)")
        cols = list(df.columns)

        date_col = st.selectbox(
            "Select Date Column",
            cols,
            index=cols.index(detected["date"]) if detected["date"] and detected["date"] in cols else 0
        )
        product_col = st.selectbox(
            "Select Product Column",
            cols,
            index=cols.index(detected["product"]) if detected["product"] and detected["product"] in cols else 0
        )
        qty_col = st.selectbox(
            "Select Quantity Column",
            cols,
            index=cols.index(detected["quantity"]) if detected["quantity"] and detected["quantity"] in cols else 0
        )
        price_col = st.selectbox(
            "Select Price Column",
            cols,
            index=cols.index(detected["price"]) if detected["price"] and detected["price"] in cols else 0
        )

        sales_col = st.selectbox("Select Sales Column (optional)", [None] + cols, index=0)

        df_clean = preprocess_data(df, date_col, product_col, qty_col, price_col, sales_col)

        st.write("Original Rows:", len(df))
        st.write("Clean Rows:", len(df_clean))

        if df_clean.empty:
            st.error("❌ After cleaning dataset became empty. Please select correct columns.")
            st.stop()

        st.success("✅ Dataset processed successfully!")
        st.dataframe(df_clean.head())

        st.download_button(
            "⬇️ Download Cleaned Dataset (CSV)",
            df_clean.to_csv(index=False).encode("utf-8"),
            "cleaned_dataset.csv",
            "text/csv"
        )

        st.session_state["df_clean"] = df_clean

    # ---------------- AFTER PROCESSING ----------------
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]

        summary = generate_summary(df_clean)
        top_df, low_df = top_low_products(df_clean)
        product_summary = product_sales_summary(df_clean)

        # ---------------- Sales Analytics ----------------
        if menu == "Sales Analytics":
            st.subheader("📈 Sales Analytics Dashboard")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Sales", f"{summary['Total Sales']:.2f}")
            col2.metric("Total Products", summary["Total Products"])
            col3.metric("Best Selling Product", summary["Best Selling Product"])
            col4.metric("Worst Selling Product", summary["Worst Selling Product"])
            col5.metric("Avg Sale", f"{summary['Average Sale per Transaction']:.2f}")
            col6.metric("Total Transactions", summary["Total Rows"])

            st.subheader("📊 Sales Trend Over Time")
            daily_sales = df_clean.groupby("Date")["Total_Sales"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(daily_sales["Date"], daily_sales["Total_Sales"])
            ax.set_title("Sales Trend")
            ax.set_xlabel("Date")
            ax.set_ylabel("Total Sales")
            st.pyplot(fig)

        # ---------------- Product Insights ----------------
        if menu == "Product Insights":
            st.subheader("🔍 Product Insights")

            st.subheader("✅ Top Selling Products")
            st.dataframe(top_df)

            st.subheader("❌ Low Selling Products")
            st.dataframe(low_df)

            st.subheader("📌 Product Sales Summary")
            st.dataframe(product_summary)

            st.subheader("📊 Top Products Bar Chart")
            fig, ax = plt.subplots(figsize=(10, 4))
            top_df.head(10).plot.bar(x="Product", y="Total_Sales", ax=ax)
            ax.set_title("Top 10 Products by Sales")
            ax.set_ylabel("Total Sales")
            st.pyplot(fig)

        # ---------------- Forecasting ----------------
        if menu == "Forecasting (Product Wise)":
            st.subheader("🤖 Product-Wise Forecasting")

            model_path = "models/product_forecast_model.pkl"

            # Load model if exists
            model, le = None, None
            if os.path.exists(model_path):
                model, le = joblib.load(model_path)

                # Check if new products exist
                trained_products = set(le.classes_)
                current_products = set(df_clean["Product"].unique())

                if not current_products.issubset(trained_products):
                    st.warning("⚠️ New products found in this CSV. Please retrain model to avoid unseen label error.")
                    model, le = None, None

            if st.button("Train Product Forecast Model"):
                model, le = train_product_forecast_model(df_clean)
                os.makedirs("models", exist_ok=True)
                joblib.dump((model, le), model_path)
                st.success("✅ Product forecast model trained & saved!")

            if model is not None and le is not None:
                future_days = st.number_input("Predict next N days", min_value=1, max_value=60, value=7)

                product_options = ["All Products"] + list(df_clean["Product"].unique())
                selected_product = st.selectbox("Select Product (or All)", product_options, index=0)

                if st.button("Predict Future Sales"):
                    selected_prod = None if selected_product == "All Products" else selected_product

                    future_df = predict_product_future_sales(df_clean, model, le, future_days, selected_prod)
                    st.dataframe(future_df)

                    if selected_prod is None and not future_df.empty:
                        top_future = get_top_future_products(future_df)
                        st.subheader("🏆 Top Future Selling Products")
                        st.dataframe(top_future)

                    if not future_df.empty:
                        st.subheader("📈 Future Sales Prediction Chart (Top 5 Products)")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        for prod in future_df["Product"].unique()[:5]:
                            prod_data = future_df[future_df["Product"] == prod]
                            ax.plot(prod_data["Date"], prod_data["Predicted_Sales"], label=prod)

                        ax.set_title("Future Sales Prediction")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Predicted Sales")
                        ax.legend()
                        st.pyplot(fig)

                    st.session_state["future_df"] = future_df
            else:
                st.info("ℹ️ Train the model first to enable forecasting.")

        # ---------------- Downloads ----------------
        if menu == "Downloads":
            st.subheader("⬇️ Download Reports & Data")

            st.download_button(
                "⬇️ Download Cleaned Dataset (CSV)",
                df_clean.to_csv(index=False).encode("utf-8"),
                "cleaned_dataset.csv",
                "text/csv"
            )

            st.download_button(
                "⬇️ Download Top Products (CSV)",
                top_df.to_csv(index=False).encode("utf-8"),
                "top_products.csv",
                "text/csv"
            )

            st.download_button(
                "⬇️ Download Low Products (CSV)",
                low_df.to_csv(index=False).encode("utf-8"),
                "low_products.csv",
                "text/csv"
            )

            if "future_df" in st.session_state:
                future_df = st.session_state["future_df"]
                st.download_button(
                    "⬇️ Download Future Predictions (CSV)",
                    future_df.to_csv(index=False).encode("utf-8"),
                    "future_predictions.csv",
                    "text/csv"
                )

            if st.button("Generate PDF Report"):
                future_df = st.session_state.get("future_df", None)
                filename = generate_pdf_report(summary, top_df, low_df, future_df, filename="company_report.pdf")

                with open(filename, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        f,
                        file_name="company_report.pdf",
                        mime="application/pdf"
                    )

else:
    st.info("👆 Upload a CSV file to start.")
