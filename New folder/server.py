import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
        # Load the CSV file
        data = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading CSV file: {e}", 500

    required_columns = {'Product_Code', 'Warehouse', 'Product_Category', 'Date', 'Order_Demand'}
    if not required_columns.issubset(data.columns):
        return f"CSV missing required columns: {required_columns - set(data.columns)}", 400

    # Data preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce').fillna(0)
    data = data.sort_values('Date')

    # Get top products by total demand
    top_products = data.groupby('Product_Code')['Order_Demand'].sum().nlargest(10).index
    top_data = data[data['Product_Code'].isin(top_products)]

    products = []

    for product in top_products:
        product_data = top_data[top_data['Product_Code'] == product]
        product_grouped = product_data.groupby('Date')['Order_Demand'].sum()
        product_grouped = product_grouped.asfreq('D', fill_value=0)

        # Check if there is enough data for the Holt-Winters model
        if len(product_grouped) < 730:  # Less than two years of daily data
            # Fallback mechanism: use simple moving average for forecasting
            print(f"Insufficient data for product {product}. Using moving average.")
            daily_forecast = product_grouped.rolling(window=30).mean().shift(-30)  # 30-day moving average as a simple forecast
            weekly_forecast = daily_forecast.resample('W').sum()
            monthly_forecast = daily_forecast.resample('M').sum()

        else:
            # Apply Holt-Winters Exponential Smoothing if sufficient data is available
            try:
                model = ExponentialSmoothing(
                    product_grouped,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=365  # Adjust this based on data
                ).fit()
                forecast_days = 30
                forecast = model.forecast(forecast_days)
                forecast_index = pd.date_range(product_grouped.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
                daily_forecast = pd.Series(forecast, index=forecast_index)

                # Weekly forecast
                weekly_forecast = daily_forecast.resample('W').sum()

                # Monthly forecast
                monthly_forecast = daily_forecast.resample('M').sum()

            except Exception as e:
                return f"Forecasting failed for product {product}: {e}", 500

        # Save forecasts as CSV
        daily_csv_path = os.path.join(OUTPUT_FOLDER, f'daily_forecast_{product}.csv')
        daily_forecast.to_csv(daily_csv_path, header=["Forecast"], index_label="Date")

        weekly_csv_path = os.path.join(OUTPUT_FOLDER, f'weekly_forecast_{product}.csv')
        weekly_forecast.to_csv(weekly_csv_path, header=["Forecast"], index_label="Date")

        monthly_csv_path = os.path.join(OUTPUT_FOLDER, f'monthly_forecast_{product}.csv')
        monthly_forecast.to_csv(monthly_csv_path, header=["Forecast"], index_label="Date")

        # Visualization for all forecasts
        plt.figure(figsize=(14, 8))
        plt.plot(product_grouped, label='Historical Data', color='blue')
        plt.plot(daily_forecast, label='Daily Forecast', color='orange')
        plt.plot(weekly_forecast, label='Weekly Forecast', color='green', linestyle='--')
        plt.plot(monthly_forecast, label='Monthly Forecast', color='red', linestyle='-.')
        plt.legend()
        plt.title(f'Forecast Visualization for Product: {product}')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')

        visualization_path = os.path.join(VISUALIZATION_FOLDER, f'visualization_{product}.png')
        plt.savefig(visualization_path)
        plt.close()

        products.append({
            "product": product,
            "daily": os.path.basename(daily_csv_path),
            "weekly": os.path.basename(weekly_csv_path),
            "monthly": os.path.basename(monthly_csv_path),
            "visualization": os.path.basename(visualization_path)
        })

    return render_template('results.html', products=products)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/visualize/<filename>')
def visualize_file(filename):
    return send_from_directory(VISUALIZATION_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
