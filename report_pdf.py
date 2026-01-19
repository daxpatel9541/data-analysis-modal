from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf_report(summary, top_df, low_df, future_df=None, filename="report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Universal Company Analyzer Report")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary:")
    y -= 20

    c.setFont("Helvetica", 11)
    for k, v in summary.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Products:")
    y -= 20

    c.setFont("Helvetica", 10)
    for _, row in top_df.head(10).iterrows():
        c.drawString(60, y, f"{row['Product']}  |  {row['Total_Sales']}")
        y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Low Products:")
    y -= 20

    c.setFont("Helvetica", 10)
    for _, row in low_df.head(10).iterrows():
        c.drawString(60, y, f"{row['Product']}  |  {row['Total_Sales']}")
        y -= 15

    if future_df is not None and not future_df.empty:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Future Sales Prediction:")
        y -= 20

        c.setFont("Helvetica", 10)
        for _, row in future_df.head(10).iterrows():
            c.drawString(60, y, f"{row['Date'].date()}  |  {round(row['Predicted_Sales'], 2)}")
            y -= 15

    c.save()
    return filename


