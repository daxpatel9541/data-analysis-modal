def auto_detect_columns(columns):
    cols = [c.lower() for c in columns]

    def find_col(keywords):
        for c in columns:
            low = c.lower()
            for k in keywords:
                if k in low:
                    return c
        return columns[0]  # fallback

    return {
        "date": find_col(["date", "order_date", "invoice_date", "time"]),
        "product": find_col(["product", "item", "name", "sku"]),
        "quantity": find_col(["qty", "quantity", "units", "count"]),
        "price": find_col(["price", "sales", "amount", "revenue", "total"])
    }
