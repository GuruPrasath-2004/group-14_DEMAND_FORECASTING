import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
OUTPUT_FOLDER = 'output'
VISUALIZATION_FOLDER = 'visualizations'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return process_file(filepath)

def process_file(filepath):
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading CSV file: {e}", 500

    required_columns = {'Product_Code', 'Warehouse', 'Product_Category', 'Date', 'Order_Demand'}
    if not required_columns.issubset(data.columns):
        return f"CSV missing required columns: {required_columns - set(data.columns)}", 400

    data['Date'] = pd.to_datetime(data['Date'])
    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce').fillna(0)
    data = data.sort_values('Date')

    top_products = data.groupby('Product_Code')['Order_Demand'].sum().nlargest(10).index
    top_data = data[data['Product_Code'].isin(top_products)]

    csv_paths = {"daily": [], "weekly": [], "monthly": []}
    visualization_paths = []

    for product in top_products:
        product_data = top_data[top_data['Product_Code'] == product]
        product_grouped = product_data.groupby('Date')['Order_Demand'].sum()
        product_grouped = product_grouped.asfreq('D', fill_value=0)

        # Daily forecast
        daily_forecast = product_grouped.resample('D').sum().tail(30)
        daily_csv_path = os.path.join(OUTPUT_FOLDER, f'daily_forecast_{product}.csv')
        daily_forecast.to_csv(daily_csv_path, header=["Forecast"], index_label="Date")
        csv_paths["daily"].append(daily_csv_path)

        # Weekly forecast
        weekly_forecast = product_grouped.resample('W').sum().tail(12)
        weekly_csv_path = os.path.join(OUTPUT_FOLDER, f'weekly_forecast_{product}.csv')
        weekly_forecast.to_csv(weekly_csv_path, header=["Forecast"], index_label="Date")
        csv_paths["weekly"].append(weekly_csv_path)

        # Monthly forecast
        monthly_forecast = product_grouped.resample('M').sum().tail(12)
        monthly_csv_path = os.path.join(OUTPUT_FOLDER, f'monthly_forecast_{product}.csv')
        monthly_forecast.to_csv(monthly_csv_path, header=["Forecast"], index_label="Date")
        csv_paths["monthly"].append(monthly_csv_path)

        # Visualization for each product
        plt.figure(figsize=(14, 8))
        plt.plot(product_grouped, label='Historical Data', color='blue')
        plt.plot(daily_forecast, label='Daily Forecast', color='orange')
        plt.plot(weekly_forecast, label='Weekly Forecast', color='green')
        plt.plot(monthly_forecast, label='Monthly Forecast', color='red')
        plt.legend()
        plt.title(f'Holt-Winters Forecast for Product: {product}')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')

        visualization_path = os.path.join(VISUALIZATION_FOLDER, f'visualization_{product}.png')
        plt.savefig(visualization_path)
        plt.close()

        visualization_paths.append(visualization_path)

    csv_links = "".join([
        f"""
        <h3>Product: {product}</h3>
        <ul>
            <li><a href='/download/{os.path.basename(daily)}'>Daily Forecast CSV</a></li>
            <li><a href='/download/{os.path.basename(weekly)}'>Weekly Forecast CSV</a></li>
            <li><a href='/download/{os.path.basename(monthly)}'>Monthly Forecast CSV</a></li>
        </ul>
        <p><a href='/visualize/{os.path.basename(visualization)}'>View Visualization</a></p>
        """
        for product, daily, weekly, monthly, visualization in zip(
            top_products, csv_paths["daily"], csv_paths["weekly"], csv_paths["monthly"], visualization_paths
        )
    ])

    return f"""
    <h1>File Processed Successfully</h1>
    {csv_links}
    """

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/visualize/<filename>')
def visualize_file(filename):
    return send_from_directory(VISUALIZATION_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
