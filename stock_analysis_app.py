#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from flair.models import TextClassifier
from flair.data import Sentence
import pytz

# Define the anchored_vwap function
def anchored_vwap(data, anchor_date):
    anchor_date = pd.Timestamp(anchor_date).tz_localize(data.index.tz)
    anchor_data = data.loc[anchor_date:]
    cumulative_volume = anchor_data['Volume'].cumsum()
    cumulative_volume_price = (anchor_data['Close'] * anchor_data['Volume']).cumsum()
    return cumulative_volume_price / cumulative_volume

# Function to perform stock analysis
def analyze_stock(ticker, period, interval, anchor_date):
    # Download data
    stock_data = yf.download(tickers=ticker, period=period, interval=interval)

    # Filter the data
    filtered = stock_data

    if filtered.empty:
        st.error("No data available for the specified date range.")
        st.info(f"Available date range: {stock_data.index[0]} to {stock_data.index[-1]}")
        return None

    # Calculate indicators
    filtered['SMA20'] = filtered['Close'].rolling(window=20).mean()
    filtered['SMA150'] = filtered['Close'].rolling(window=150).mean()
    filtered['EMA20'] = filtered['Close'].ewm(span=20, adjust=False).mean()
    filtered['EMA150'] = filtered['Close'].ewm(span=150, adjust=False).mean()
    filtered['VWAP'] = (filtered['Close'] * filtered['Volume']).cumsum() / filtered['Volume'].cumsum()

    # Calculate Anchored VWAP
    filtered['Anchored_VWAP'] = anchored_vwap(filtered, anchor_date)

    # Normalize volume (0 to 1 scale)
    filtered['NormalizedVolume'] = (filtered['Volume'] - filtered['Volume'].min()) / (filtered['Volume'].max() - filtered['Volume'].min())

    return filtered

# Function to create the plot
def create_plot(filtered, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=filtered.index,
                                 open=filtered['Open'],
                                 high=filtered['High'],
                                 low=filtered['Low'],
                                 close=filtered['Close'],
                                 name='Price'),
                  row=1, col=1)

    # Add indicators
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['SMA20'], name='SMA20', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['SMA150'], name='SMA150', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['EMA20'], name='EMA20', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['EMA150'], name='EMA150', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['VWAP'], name='VWAP', line=dict(color='magenta', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered['Anchored_VWAP'], name='Anchored VWAP', line=dict(color='cyan', dash='dot')), row=1, col=1)

    # Add volume chart
    colors = ['red' if close < open else 'blue' for close, open in zip(filtered['Close'], filtered['Open'])]
    fig.add_trace(go.Bar(x=filtered.index, y=filtered['NormalizedVolume'], name='Volume', marker_color=colors), row=2, col=1)

    # Update layout
    fig.update_layout(title=f'{ticker} Analysis',
                      yaxis_title='Price',
                      yaxis2_title='Normalized Volume',
                      xaxis_rangeslider_visible=False,
                      height=800,
                      showlegend=True)

    # Update y-axis of volume subplot to fixed range
    fig.update_yaxes(range=[0, 1], row=2, col=1)

    return fig

# Function to perform news sentiment analysis
def analyze_news_sentiment(ticker, start_date, end_date):
    newsapi = NewsApiClient(api_key='a64474fc27b6485294e6e08e893797d8')
    
    response = newsapi.get_everything(
        q=ticker,
        from_param=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d'),
        language="en",
    )

    model = TextClassifier.load('sentiment')
    sentiment = 0
    articles = []

    for article in response['articles']:
        title = article['title']
        url = article['url']
        sentence = Sentence(title)
        model.predict(sentence)
        total_sentiment = sentence.labels

        if total_sentiment[0].value == 'NEGATIVE':
            sentiment -= total_sentiment[0].to_dict()['confidence'] / 2
        elif total_sentiment[0].value == 'POSITIVE':
            sentiment += total_sentiment[0].to_dict()['confidence']

        articles.append({
            'title': title,
            'url': url,
            'sentiment': total_sentiment[0].value,
            'confidence': total_sentiment[0].to_dict()['confidence']
        })

    return sentiment, articles

# Streamlit app
def main():
    st.title("Stock Analysis App")

    # User input for stock ticker
    ticker = st.text_input("Enter stock ticker (e.g., TSLA):", "TSLA").upper()

    # User input for data period
    period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    period = st.selectbox("Select data period:", period_options, index=period_options.index('1y'))

    # User input for data interval
    interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    interval = st.selectbox("Select data interval:", interval_options, index=interval_options.index('1d'))

    # User input for anchored VWAP date
    default_anchor_date = (datetime.now(pytz.utc) - timedelta(days=180)).date()
    anchor_date = st.date_input("Select anchored VWAP date:", value=default_anchor_date)

    # User input for news date range
    st.subheader("News Analysis Date Range")
    col1, col2 = st.columns(2)
    with col1:
        news_start_date = st.date_input("Start date", value=datetime.now(pytz.utc).date() - timedelta(days=7))
    with col2:
        news_end_date = st.date_input("End date", value=datetime.now(pytz.utc).date())

    if st.button("Analyze"):
        with st.spinner("Analyzing stock data..."):
            filtered_data = analyze_stock(ticker, period, interval, anchor_date)

        if filtered_data is not None:
            st.success("Analysis complete!")

            # Display the plot
            st.plotly_chart(create_plot(filtered_data, ticker))

            # Display date range of plotted data
            st.info(f"Date range of plotted data: {filtered_data.index[0]} to {filtered_data.index[-1]}")

            # Perform news sentiment analysis
            with st.spinner("Analyzing news sentiment..."):
                sentiment_score, articles = analyze_news_sentiment(ticker, news_start_date, news_end_date)

            st.subheader("News Sentiment Analysis")
            st.write(f"Total Sentiment Score: {sentiment_score:.2f}")
            st.write(f"News date range: {news_start_date} to {news_end_date}")

            st.subheader("Recent News Articles")
            for article in articles:
                st.write(f"{article['title']} - Sentiment: {article['sentiment']} ({article['confidence']:.2f})")
                st.write(f"Read more: {article['url']}")
                st.write("---")

if __name__ == "__main__":
    main()
