def auto_detect_columns(columns):
    def find_col(keywords):
        for c in columns:
            low = c.lower()
            for k in keywords:
                if k in low:
                    return c
        return None

    date_col = find_col(["date", "order_date", "invoice_date", "time"])
    product_col = find_col(["product", "item", "name", "sku"])
    quantity_col = find_col(["qty", "quantity", "units", "count"])
    price_col = find_col(["price", "unit_price", "cost"])
    sales_col = find_col(["total_sales", "sales", "amount", "revenue", "total"])

    return {
        "date": date_col,
        "product": product_col,
        "quantity": quantity_col,
        "price": price_col,
        "sales": sales_col
    }


