import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to run the strategy
def run_strategy(data, stop_loss_pct=None):
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    data['Signal'] = np.where(data['Close'] < 0.97 * data['SMA_20'], 1, 0)
    data['Position'] = 0
    entry_price = 0

    for i in range(1, len(data)):
        if data['Signal'].iloc[i-1] == 1 and data['Position'].iloc[i-1] == 0:
            entry_price = data['Close'].iloc[i]
            data.iloc[i, data.columns.get_loc('Position')] = 1
        elif data['Position'].iloc[i-1] == 1:
            if stop_loss_pct and data['Close'].iloc[i] < (1 - stop_loss_pct) * entry_price:
                data.iloc[i, data.columns.get_loc('Position')] = 0
                entry_price = 0
            else:
                data.iloc[i, data.columns.get_loc('Position')] = 1
        else:
            data.iloc[i, data.columns.get_loc('Position')] = 0

    data['Strategy_Return'] = data['Position'].shift(1).fillna(0) * data['Return']
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    return data

# Download data
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-12-31'
raw_data = yf.download(ticker, start=start_date, end=end_date)

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

# Market performance
raw_data['Return'] = raw_data['Close'].pct_change()
raw_data['Cumulative_Market'] = (1 + raw_data['Return']).cumprod()

# Strategy A: With stop-loss
strategy_with_sl = run_strategy(raw_data, stop_loss_pct=0.05)

# Strategy B: Without stop-loss
strategy_without_sl = run_strategy(raw_data, stop_loss_pct=None)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(raw_data['Cumulative_Market'], label='Market (Buy & Hold)', color='blue')
plt.plot(strategy_without_sl['Cumulative_Strategy'], label='Mean Reversion (No Stop-Loss)', color='orange')
plt.plot(strategy_with_sl['Cumulative_Strategy'], label='Mean Reversion (Stop-Loss)', color='green')
plt.title(f'{ticker} - Mean Reversion Strategy Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Performance summary
def performance_summary(data, name):
    strat_ret = data['Strategy_Return']
    total_return = data['Cumulative_Strategy'].iloc[-1] - 1
    sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std()
    drawdown = (data['Cumulative_Strategy'] / data['Cumulative_Strategy'].cummax() - 1).min()
    return {
        'Strategy': name,
        'Total Return': f"{total_return:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{drawdown:.2%}"
    }

summary_with = performance_summary(strategy_with_sl, "With Stop-Loss")
summary_without = performance_summary(strategy_without_sl, "No Stop-Loss")

# Print summary table
summary_df = pd.DataFrame([summary_without, summary_with])
print("\n📊 Strategy Performance Comparison:")
print(summary_df)
