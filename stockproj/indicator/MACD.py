# -*- coding: utf-8 -*-
"""
Refactored on Mon Sep 1 2025

Refactored version with added MACD plotting functionality.
Contains separate functions for calculations and plotting.
"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_MACD(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates MACD for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "close": Closing price
        fast_period (int): The period for the fast EMA (default is 12).
        slow_period (int): The period for the slow EMA (default is 26).
        signal_period (int): The period for the signal line EMA (default is 9).

    Returns:
        pd.DataFrame: DataFrame with columns "MACD" and "Signal".
    """
    ema_fast = data["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow_period, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()

    return pd.DataFrame({"MACD": macd, "Signal": signal})

def plot_MACD(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Plots the MACD chart.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data with columns:
            - "time": Timestamp
            - "close": Closing price
        fast_period (int): The period for the fast EMA (default is 12).
        slow_period (int): The period for the slow EMA (default is 26).
        signal_period (int): The period for the signal line EMA (default is 9).

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

        macd_data = calculate_MACD(data, fast_period, slow_period, signal_period)
        data = pd.concat([data, macd_data], axis=1)

        fig = go.Figure()

        # Add MACD trace
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="blue")
        ))

        # Add Signal trace
        fig.add_trace(go.Scatter(
            x=data["time"],
            y=data["Signal"],
            mode="lines",
            name="Signal",
            line=dict(color="red")
        ))

        # Add histogram
        fig.add_trace(go.Bar(
            x=data["time"],
            y=(data["MACD"] - data["Signal"]),
            name="Histogram",
            marker_color="gray"
        ))

        fig.update_layout(
            title="MACD Chart",
            xaxis=dict(showline=True, linecolor="#888"),
            yaxis=dict(showline=True, linecolor="#888"),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        return fig

    except ValueError as e:
        st.error(f"Error while plotting MACD chart: {str(e)}")
        return None

# Example usage
# Assuming `data` is a DataFrame containing "time" and "close" columns
# fig = plot_MACD(data)
# fig.show()