import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_ATR(data, timeframe=14):
    """
    Calculates ATR for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "high": Highest price
            - "low": Lowest price
            - "close": Closing price
        timeframe (int): The timeframe for ATR calculation (default is 14).

    Returns:
        pd.Series: ATR values.
    """
    high_low = (data["high"] - data["low"])/data["low"]
    high_close = ((data["high"] - data["close"].shift(1))/data["close"].shift(1)).abs()
    low_close = ((data["low"] - data["close"].shift(1))/data["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=timeframe, min_periods=timeframe).mean()
    return atr*100  # Convert to percentage

def plot_ATR(data, timeframe=14):
    """
    Plots the ATR chart.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "time": Timestamp
            - "high": Highest price
            - "low": Lowest price
            - "close": Closing price
        timeframe (int): The timeframe for ATR calculation (default is 14).

    Returns:
        plotly.graph_objects.Figure: The plotted figure.
    """
    try:
        required_columns = {"time", "high", "low", "close"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data["time"] = pd.to_datetime(data["time"])
        data = data.sort_values("time").reset_index(drop=True)

        atr = calculate_ATR(data, timeframe)
        data["ATR_14"] = atr

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["ATR_14"].fillna(0),
            mode="lines",
            name="ATR_14",
            line=dict(color="purple")
        ))

        fig.update_layout(
            title="ATR Chart",
            xaxis=dict(showline=True, linecolor="#888"),
            yaxis=dict(showline=True, linecolor="#888"),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        return fig

    except ValueError as e:
        st.error(f"Error while plotting ATR chart: {str(e)}")
        return None

def create_scatter_plot(data, timeframe=14):
    """
    create_scatter_plot the ATR chart.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "time": Timestamp
            - "high": Highest price
            - "low": Lowest price
            - "close": Closing price
        timeframe (int): The timeframe for ATR calculation (default is 14).

    Returns:
        plotly.graph_objects.Figure: The plotted figure.
    """
    try:
        required_columns = {"time", "high", "low", "close"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data["time"] = pd.to_datetime(data["time"])
        data = data.sort_values("time").reset_index(drop=True)

        atr = calculate_ATR(data, timeframe)
        data["ATR_14"] = atr

        go_scatter = go.Scatter(
            x=data["time"],
            y=data["ATR_14"].fillna(0),
            mode="lines",
            name="ATR_14",
            line=dict(color="purple")
        )
        return go_scatter

    except ValueError as e:
        st.error(f"Error while plotting ATR chart: {str(e)}")
        return None