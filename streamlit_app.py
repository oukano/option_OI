import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# Black-Scholes Gamma Calculation Function with error handling
def calculate_gamma(S, K, T, r, sigma):
    # Avoid division by zero or invalid values
    if K == 0 or sigma == 0 or T == 0:
        return 0  # Return zero gamma for invalid values
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except (ZeroDivisionError, ValueError):
        return 0  # Handle any other edge cases like division by zero or math errors

# Automatically refresh the app every 60 seconds
st_autorefresh(interval=60 * 1000)  # 60 seconds

# Streamlit app
st.title("Options Analysis Tool")

# Dropdown for ticker selection
ticker_options = {
    "NVDA": "NVIDIA",
    "QQQ": "Nasdaq",
    "SPY": "S&P"
}

# Ticker selection
selected_ticker = st.selectbox("Select a Ticker", list(ticker_options.keys()))

# Number of top strikes input
num_strikes = st.number_input("Select Number of Top Strikes", min_value=1, max_value=10, value=5, step=1)

# Fetch selected ticker's option chain
ticker = selected_ticker
st.subheader(f"Analyzing Options for {ticker_options[selected_ticker]} ({selected_ticker})")
nvda = yf.Ticker(ticker)

# Get current stock price
current_price = nvda.history(period="1d")['Close'].iloc[0]

# Get today's date options (0DTE) by selecting the nearest expiry date
options = nvda.options
expiry_date = options[0]
option_chain = nvda.option_chain(expiry_date)

# Get calls and puts
calls = option_chain.calls
puts = option_chain.puts

# Calculate gamma for each option using the Black-Scholes formula
risk_free_rate = 0.05  # 5% risk-free rate
time_to_expiration = 1 / 365  # 0DTE means the expiration is in one day

# Calculate gamma for calls and puts
calls['gamma_manual'] = calls.apply(
    lambda x: calculate_gamma(
        S=current_price, 
        K=x['strike'], 
        T=time_to_expiration, 
        r=risk_free_rate, 
        sigma=x['impliedVolatility']
    ), axis=1
)

puts['gamma_manual'] = puts.apply(
    lambda x: calculate_gamma(
        S=current_price, 
        K=x['strike'], 
        T=time_to_expiration, 
        r=risk_free_rate, 
        sigma=x['impliedVolatility']
    ), axis=1
)

# Aggregate open interest by strike
calls_oi = calls.groupby('strike').agg({'openInterest': 'sum', 'gamma_manual': 'sum'}).reset_index()
calls_oi.rename(columns={'openInterest': 'calls_openInterest', 'gamma_manual': 'calls_gamma'}, inplace=True)

puts_oi = puts.groupby('strike').agg({'openInterest': 'sum', 'gamma_manual': 'sum'}).reset_index()
puts_oi.rename(columns={'openInterest': 'puts_openInterest', 'gamma_manual': 'puts_gamma'}, inplace=True)

# Merge calls and puts open interest and gamma
combined_oi = pd.merge(calls_oi, puts_oi, on='strike', how='outer').fillna(0)

# Calculate total open interest
combined_oi['total_open_interest'] = combined_oi['calls_openInterest'] + combined_oi['puts_openInterest']

# Find the top N strikes with the highest open interest (based on user input)
top_strikes = combined_oi.nlargest(num_strikes, 'total_open_interest')

# Fetch intraday 15-minute stock price data for the last 7 days
nvda_intraday = nvda.history(period="7d", interval="15m")

# Plotting intraday data for one week
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(nvda_intraday.index, nvda_intraday['Close'], label=f"{ticker} 15-min Interval Price (1 Week)", color="blue")
ax.set_title(f'{ticker} One Week’s 15-Minute Stock Price with Top {num_strikes} Open Interest Strikes')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.yaxis.tick_right()  # Move y-axis ticks to the right
ax.yaxis.set_label_position("right")  # Move y-axis label to the right

# Get the min and max from both stock price and top strike prices for dynamic y-axis
min_price = min(nvda_intraday['Close'].min(), top_strikes['strike'].min())
max_price = max(nvda_intraday['Close'].max(), top_strikes['strike'].max())

# Set y-axis dynamically based on the stock price and strike price
ax.set_ylim(min_price * 0.95, max_price * 1.05)  # adding 5% buffer

# Mark the top strike prices on the chart and label them
for i, row in top_strikes.iterrows():
    strike_price = row['strike']
    ax.axhline(y=strike_price, color='red', linestyle='--', label=f"Strike {strike_price} (OI: {int(row['calls_openInterest'] + row['puts_openInterest'])})")
    ax.text(nvda_intraday.index[-1], strike_price, f"{strike_price}", color='red', verticalalignment='bottom', fontsize=10)

# Add legend to the plot
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)

# Output the top strikes and their total open interest, gamma, calls, and puts as a table
st.subheader(f"Top {num_strikes} Strikes with Highest Open Interest:")

# Calculate percentages for calls and puts
top_strikes['calls_percentage'] = (top_strikes['calls_openInterest'] / top_strikes['total_open_interest']) * 100
top_strikes['puts_percentage'] = (top_strikes['puts_openInterest'] / top_strikes['total_open_interest']) * 100

# Create a DataFrame for display
display_data = top_strikes[['strike', 'total_open_interest', 'calls_percentage', 'puts_percentage', 'calls_gamma', 'puts_gamma']].copy()
display_data['Total Gamma'] = display_data['calls_gamma'] + display_data['puts_gamma']

# Highlight the row with the closest price to the current price
def highlight_closest_price(row):
    return ['background-color: yellow' if abs(row['strike'] - current_price) == abs(display_data['strike'] - current_price).min() else '' for _ in row]

# Display the DataFrame as a Streamlit table with highlighting
st.table(display_data.style.apply(highlight_closest_price, axis=1).format({
    'strike': "{:.1f}",
    'total_open_interest': "{:.1f}",
    'calls_percentage': "{:.2f}%",
    'puts_percentage': "{:.2f}%",
    'Total Gamma': "{:.6f}"
}))
