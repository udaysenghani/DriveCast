import pandas as pd
from prophet import Prophet
import os

# Load cleaned dataset
df = pd.read_csv("./data/cleaned_dataset.csv")

# Ensure correct column names
if "date" not in df.columns or "monthly_value" not in df.columns or "english_name" not in df.columns:
    raise ValueError("Dataset must have 'date', 'monthly_value', and 'english_name' columns")

results = []

# Loop through each KPI and forecast next 3 months
for kpi in df['english_name'].unique():
    kpi_data = df[df['english_name'] == kpi][['date','monthly_value']].rename(columns={'date':'ds','monthly_value':'y'})
    
    if len(kpi_data) < 6:  # skip very short series
        continue
    
    model = Prophet(yearly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # captures monthly patterns
    model.fit(kpi_data)
    
    # Make future dataframe (3 months ahead)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    
    # Store results
    forecast['english_name'] = kpi
    results.append(forecast[['ds','yhat','yhat_lower','yhat_upper','english_name']])

# Combine all KPI forecasts
pred_df = pd.concat(results)

pred_df.to_csv("predictions/kpi_forecasts.csv", index=False)

print("✅ Forecasts saved at: predictions/kpi_forecasts.csv")
