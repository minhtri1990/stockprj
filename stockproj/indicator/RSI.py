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

def plot_RSI(data, timeframe=14):
    """
    Plots the RSI chart.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "time": Timestamp
            - "close": Closing price
        timeframe (int): The timeframe for RSI calculation (default is 14).

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

        rsi = calculate_RSI(data, timeframe)
        data["RSI"] = rsi

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="blue")
        ))

        for level, dash in zip([20, 30, 70, 80], ["dash", "solid", "solid", "dash"]):
            fig.add_hline(y=level, line_dash=dash, line_color="gray")

        fig.update_layout(
            title="RSI Chart",
            xaxis=dict(showline=True, linecolor="#888"),
            yaxis=dict(showline=True, linecolor="#888"),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        return fig

    except ValueError as e:
        st.error(f"Error while plotting RSI chart: {str(e)}")
        return None


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