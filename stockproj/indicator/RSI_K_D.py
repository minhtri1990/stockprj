import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_RSI(data, timeframe=14):
    """
    Calculates RSI for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "close": Closing price
        timeframe (int): The timeframe for RSI calculation (default is 14).

    Returns:
        pd.Series: RSI values.
    """
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=timeframe, min_periods=timeframe).mean()
    avg_loss = loss.rolling(window=timeframe, min_periods=timeframe).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_RSI(data, rsi_window=14, k_window=3, d_window=3):
    """
    Calculates Stochastic RSI for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "close": Closing price
        rsi_window (int): The window size for RSI calculation (default is 14).
        k_window (int): The window size for %K smoothing (default is 3).
        d_window (int): The window size for %D smoothing (default is 3).

    Returns:
        pd.DataFrame: DataFrame with columns "K" and "D".
    """
    rsi = calculate_RSI(data, timeframe=rsi_window)
    min_val = rsi.rolling(window=rsi_window, center=False).min()
    max_val = rsi.rolling(window=rsi_window, center=False).max()
    stoch = ((rsi - min_val) / (max_val - min_val)) * 100

    k = stoch.rolling(window=k_window, center=False).mean()
    d = k.rolling(window=d_window, center=False).mean()

    return pd.DataFrame({"K": k, "D": d})

def plot_stoch_RSI(data, rsi_window=14, k_window=3, d_window=3):
    """
    Plots the Stochastic RSI chart.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "time": Timestamp
            - "close": Closing price
        rsi_window (int): The window size for RSI calculation (default is 14).
        k_window (int): The window size for %K smoothing (default is 3).
        d_window (int): The window size for %D smoothing (default is 3).

    Returns:
        plotly.graph_objects.Figure: The plotted figure.
    """
    try:
        required_columns = {"time", "close"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data["time"] = pd.to_datetime(data["time"])
        data = data.sort_values("time").reset_index(drop=True)

        stoch_rsi = calculate_stoch_RSI(data, rsi_window, k_window, d_window)
        data = pd.concat([data, stoch_rsi], axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["K"],
            mode="lines",
            name="RSI_K",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["D"],
            mode="lines",
            name="RSI_D",
            line=dict(color="green")
        ))

        for level, dash in zip([20, 80], ["solid", "solid"]):
            fig.add_hline(y=level, line_dash=dash, line_color="gray")

        fig.update_layout(
            title="Stochastic RSI Chart",
            xaxis=dict(showline=True, linecolor="#888"),
            yaxis=dict(showline=True, linecolor="#888"),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        return fig

    except ValueError as e:
        st.error(f"Error while plotting Stochastic RSI chart: {str(e)}")
        return None

