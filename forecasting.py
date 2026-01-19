import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(df):
    daily = df.groupby("Date")["Total_Sales"].sum().reset_index()
    daily["DayIndex"] = range(len(daily))

    X = daily[["DayIndex"]]
    y = daily["Total_Sales"]

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future_sales(df, model, future_days):
    daily = df.groupby("Date")["Total_Sales"].sum().reset_index()
    last_index = len(daily)

    future_index = list(range(last_index, last_index + future_days))
    future_dates = pd.date_range(daily["Date"].max(), periods=future_days + 1)[1:]

    preds = model.predict(pd.DataFrame({"DayIndex": future_index}))

    return pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})
