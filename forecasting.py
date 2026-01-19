import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def train_product_forecast_model(df):
    """
    Trains ONE RandomForest model for product-wise sales forecasting.
    Uses Date + Product grouped daily sales and feature engineering.
    Returns trained model and LabelEncoder.
    """

    # Group by Date and Product (daily product sales)
    product_daily = df.groupby(["Date", "Product"])["Total_Sales"].sum().reset_index()
    product_daily = product_daily.sort_values(["Product", "Date"]).reset_index(drop=True)

    # Feature Engineering
    product_daily["DayIndex"] = product_daily.groupby("Product").cumcount()
    product_daily["DayOfWeek"] = product_daily["Date"].dt.dayofweek
    product_daily["Month"] = product_daily["Date"].dt.month

    # Encode Product
    le = LabelEncoder()
    product_daily["Product_Encoded"] = le.fit_transform(product_daily["Product"])

    # Training data
    X = product_daily[["DayIndex", "DayOfWeek", "Month", "Product_Encoded"]]
    y = product_daily["Total_Sales"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model, le


def predict_product_future_sales(df, model, le, future_days, selected_product=None):
    """
    Predicts future sales for:
    - All products (if selected_product is None)
    - OR a single product (if selected_product is given)

    IMPORTANT CHANGE:
    Future dates will start from TOMORROW (real-time), not from dataset last date.
    """

    products = df["Product"].unique() if selected_product is None else [selected_product]
    all_predictions = []

    # Encoder known products
    encoder_products = set(le.classes_)

    # Real-time start date (tomorrow)
    start_date = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=future_days)

    for product in products:
        # Skip unseen product labels (avoid crash)
        if product not in encoder_products:
            continue

        product_data = df[df["Product"] == product]
        if product_data.empty:
            continue

        # Daily sales for this product (for DayIndex)
        daily = product_data.groupby("Date")["Total_Sales"].sum().reset_index()
        daily = daily.sort_values("Date").reset_index(drop=True)

        last_index = len(daily)
        future_index = list(range(last_index, last_index + future_days))

        # Create future dataframe
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Product": product
        })

        future_df["DayIndex"] = future_index
        future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek
        future_df["Month"] = future_df["Date"].dt.month
        future_df["Product_Encoded"] = le.transform([product] * len(future_df))

        # Predict
        preds = model.predict(future_df[["DayIndex", "DayOfWeek", "Month", "Product_Encoded"]])
        future_df["Predicted_Sales"] = preds

        all_predictions.append(future_df[["Date", "Product", "Predicted_Sales"]])

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Date", "Product", "Predicted_Sales"])


def get_top_future_products(pred_df, n=10):
    """
    Returns Top N products based on total predicted sales.
    """

    if pred_df.empty:
        return pd.DataFrame(columns=["Product", "Total_Predicted_Sales"])

    top_products = (
        pred_df.groupby("Product")["Predicted_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    top_products.columns = ["Product", "Total_Predicted_Sales"]
    return top_products
