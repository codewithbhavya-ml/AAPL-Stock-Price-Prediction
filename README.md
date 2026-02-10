📈 AAPL Stock Price Prediction – 30 Day Forecast

An end-to-end time series forecasting project that predicts Apple Inc. (AAPL) stock prices for the next 30 trading days using an ARIMAX model, deployed through an interactive Streamlit dashboard.

🔍 Project Overview

This project analyzes historical Apple stock data and forecasts future closing prices using ARIMAX (AutoRegressive Integrated Moving Average with Exogenous variables).
Trading volume is used as an external factor to improve forecast accuracy.
The trained model is serialized using pickle and deployed with Streamlit for real-time interaction and visualization.

🧠 Key Features

Time series forecasting using ARIMAX

Incorporation of trading volume as an exogenous variable

30-day stock price prediction

Interactive Streamlit dashboard

Visualization of historical vs forecasted prices

End-to-end workflow: EDA → Modeling → Deployment

🛠️ Tech Stack

Python, Pandas, NumPy, Matplotlib, Statsmodels (ARIMAX), Streamlit, Pickle

📂 Project Structure
├── AAPL.csv                         # Historical stock data
├── Apple_Stock_Price_Prediction.ipynb  # EDA & model training
├── best_arimax.pkl                  # Trained ARIMAX model
├── main.py                          # Streamlit app
└── README.md

📊 Model Details

Model: ARIMAX

Target Variable: Closing Price

Exogenous Variable: Trading Volume

Forecast Horizon: 30 Trading Days

The model captures historical price patterns while accounting for the influence of trading volume.

📈 Evaluation Approach

Time-based train–test split

Stationarity and residual analysis

Visual comparison of predicted vs actual prices

These steps ensure stable and reliable forecasts.

🧠 Learning Outcomes

Practical experience with time series analysis

Handling exogenous variables in forecasting models

Financial data analysis and visualization

Model deployment using Streamlit

Building production-ready data science projects

⚙️ Future Enhancements

Add technical indicators and macroeconomic variables

Experiment with LSTM and Prophet models

Improve evaluation using RMSE and MAE

Cloud deployment (Streamlit Cloud / AWS)

⚠️ Disclaimer

This project is for educational purposes only and should not be used for real investment decisions.

