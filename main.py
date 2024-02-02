from fastapi import FastAPI, HTTPException
import pandas as pd
from prophet import Prophet

# Read the CSV file and perform Prophet forecasting
def get_forecasted_data():
    data = pd.read_csv('water_usage_dataset.csv')
    df_train_prophet = data[['Date', 'Water Usage']]
    df_train_prophet = df_train_prophet.rename(columns={"Date": "ds", "Water Usage": "y"})
    m = Prophet()
    m.fit(df_train_prophet)
    future = m.make_future_dataframe(periods=365)  # 12 months
    forecast_prophet = m.predict(future)
    return forecast_prophet

# Initialize FastAPI app
app = FastAPI()

# Endpoint to get forecasted data for a given date
@app.get("/forecast/{date}")
async def get_forecast(date: str):
    try:
        # Parse the date and convert to pandas Timestamp
        date = pd.Timestamp(date)
        # Get forecasted data
        forecast_data = get_forecasted_data()
        # Filter data for the requested date
        selected_row = forecast_data[forecast_data['ds'] == date]
        if not selected_row.empty:
            return selected_row.to_dict(orient='records')[0]
        else:
            raise HTTPException(status_code=404, detail="No prediction available for the selected date.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please provide date in YYYY-MM-DD format.")