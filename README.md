# 🚗 DriveCast – KPI Forecasting & Analytics
🚗 DriveCast – A Streamlit-based KPI forecaster &amp; what-if simulator for car dealerships.  Forecast sales, revenue &amp; profit with SARIMAX, visualize correlations, and simulate business scenarios.
DriveCast is a data-driven forecasting and analytics tool that helps you analyze, visualize, and predict **business KPIs**.  
It supports **time-series forecasting** (SARIMAX & Prophet), **correlation analysis**, and a **what-if simulator** to explore how changes in one KPI impact others.

---

## 📂 Project Structure

```

DriveCast/
│── Data
│   ├── Cleaned\_dataset       # Processed data ready for forecasting
│   └── Raw\_dataset           # Original raw input data
│
│── model
│   ├── correlation.py        # Correlation analysis between KPIs
│   ├── forecast\_Prophet.py   # Forecasting with Prophet
│   └── forecast\_SARIMAX.py   # Forecasting with SARIMAX
│
│── predictions               # Stores forecast results & exports
│
│── app.py                    # Main Streamlit dashboard app
│── clean\_data.py             # Script for cleaning raw data
│── requirements.txt          # Python dependencies
│── sampledata\_2023           # Example dataset
│── README.md                 # Project documentation

````

---

## ✨ Features

- 📈 **Forecasting Models**  
  - **SARIMAX** (automatic grid search for parameters, seasonal naive fallback)  
  - **Prophet** (Facebook Prophet for trend/seasonality forecasting)  

- 📊 **Correlation Analysis**  
  - KPI correlation heatmaps  
  - Identify top positive & negative correlations  

- 🔄 **What-If Simulator**  
  - Apply a % change to a driver KPI  
  - Instantly see impacts on dependent KPIs (via log-log elasticities)  

- 🔮 **Predictions Module**  
  - Save/export forecasts to Excel  
  - Store prediction runs under `/predictions/`  

---

## 🖥️ Demo Screenshots

### Forecast_Panel 
![Dashboard Screenshot](demo_img/Forecast_Panel.png)  

### Correlation_heatmap 
![Dashboard Screenshot](demo_img/Correlation_heatmap.png)  

### What-If-Simulator  
![Forecast Screenshot](demo_img/What-If-Simulator.png)  


---

## 🚀 Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/DriveCast.git
   cd DriveCast
    ```

2. **Create a virtual environment (recommended)**

   ``` bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 📊 Usage Guide

1. Upload your **raw dataset** (CSV/Excel).
2. Run `clean_data.py` or use the in-app cleaning pipeline.
3. Choose a **forecasting model**: SARIMAX or Prophet.
4. Generate and visualize **forecasts, correlations, and what-if simulations**.
5. Export predictions to Excel under `/predictions/`.

---

## 🛠️ Tech Stack

* **Python** (pandas, numpy, statsmodels, prophet, matplotlib, seaborn)
* **Streamlit** – interactive web dashboard
* **Excel/CSV exports** for offline analysis

---

## 👨‍💻 Author

**Uday Senghani**
Software Engineer | AI & Data Enthusiast 🚀

🔗 [LinkedIn](https://linkedin.com/in/uday-senghani) 

```
