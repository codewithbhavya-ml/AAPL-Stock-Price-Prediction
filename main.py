import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AAPL Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>
.metric-card {
    background-color: #0e1117;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #9ba1a6;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Header
#---------------------------------------------------
st.title("📈 AAPL Stock Price Forecast Dashboard")
st.caption("ARIMAX-based Historical Analysis & 30-Day Forecast")

st.divider()


# Load ARIMAX Model
#---------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_arimax.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()


# Sidebar
# ----------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Adjust date range & forecast inputs")

start_date = st.sidebar.date_input(
    "📅 Start Date", pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "📅 End Date", pd.to_datetime("today")
)

st.sidebar.divider()
st.sidebar.markdown("📌 **Model:** ARIMAX")
st.sidebar.markdown("📂 **Data:** AAPL.csv")


# Load Data
# ----------------------------------------------------
@st.cache_data
def load_data(start, end):
    df = pd.read_csv("AAPL.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    df = df[["Date", "Close", "Volume"]].dropna()
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    return df.loc[start:end]

data = load_data(start_date, end_date)

if data.empty:
    st.error("❌ No data available for the selected date range.")
    st.stop()


# KPI METRICS
# ----------------------------------------------------
latest_price = data["Close"].iloc[-1]
prev_price = data["Close"].iloc[-2]
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Latest Close Price", f"${latest_price:,.2f}")

with col2:
    st.metric(
        "📈 Daily Change",
        f"{price_change:,.2f}",
        f"{pct_change:.2f}%"
    )

with col3:
    st.metric("📊 Data Points", len(data))

st.divider()

# Historical Chart
# ----------------------------------------------------
st.subheader("📊 Historical Stock Price")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(data.index, data["Close"], linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(alpha=0.3)
st.pyplot(fig)


# Forecast Inputs
# ----------------------------------------------------
st.divider()
st.subheader("🔮 30-Day Price Forecast")

default_volume = float(data["Volume"].iloc[-1])

volume_input = st.number_input(
    "Expected Daily Trading Volume",
    min_value=0.0,
    value=default_volume,
    step=1_000_000.0
)


# Forecast
# ----------------------------------------------------
if st.button("🚀 Generate Forecast", use_container_width=True):
    try:
        steps = 30
        future_exog = np.full((steps, 1), volume_input)

        forecast = model.get_forecast(steps=steps, exog=future_exog)
        forecast_values = forecast.predicted_mean

        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq="B"
        )

        forecast_df = pd.DataFrame(
            {"Predicted Close Price": forecast_values.values},
            index=future_dates
        )

        avg_forecast = forecast_df["Predicted Close Price"].mean()

        st.success("✅ Forecast generated successfully")

        # KPI for forecast
        st.metric("📈 Avg Forecast Price (30 Days)", f"${avg_forecast:,.2f}")

        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

        st.subheader("📈 Forecast vs Historical")

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(data.index, data["Close"], label="Historical", linewidth=2)
        ax2.plot(
            forecast_df.index,
            forecast_df["Predicted Close Price"],
            linestyle="--",
            linewidth=2,
            label="Forecast"
        )
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


st.divider()
st.caption("📌 Built with Streamlit | ARIMAX Model|")
