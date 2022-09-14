# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 00:10:43 2022

@author: ksuja
"""

import streamlit as st
import pandas as pd
from numpy import *
import math

import matplotlib.pyplot as plt
from tabulate import tabulate
from mibian import BS
import yfinance as yf
import cufflinks as cf
import numpy as np
import nsepy
import datetime
import glob

st.title('Stock Trend Predictor')
# Set the start and end date
start_date = '2019-01-01'
end_date = '2022-08-31'

st.image('bear-vs-bull.png')

stock = st.text_input("Enter the symbol of Stock for technical analysis")
d='.NS'
# Set the ticker
ticker = stock + d
try:
# Get the data
    data2 = yf.download(ticker, start_date, end_date)
    data = yf.download(ticker, start_date, end_date)
    data3 = yf.download(ticker, start_date, end_date)
    df2= data2
    df1=data
    df5=data3
    
    #dheeraj ema start
    df1.reset_index(drop= False, inplace=True)
    
    import pandas_ta as ta
    df1["EMA200"] = ta.ema(df1["Close"], length=200)
    df1["EMA50"] = ta.ema(df1["Close"], length=50)
    
       

    #dheeraj ema end
    short_period = 12
    long_period  = 26
    signal_period = 9
    st.text('**MACD Indicator**')
    ewm_short=data2['Adj Close'].ewm(span=short_period, adjust=False).mean()
    ewm_long=data2['Adj Close'].ewm(span=long_period, adjust=False).mean()
    MACD=ewm_short-ewm_long
    df2["MACD"]=MACD
    signal_MACD=MACD.ewm(span=signal_period, adjust=False).mean()
    df2["signal_macd"]=signal_MACD
    bars=MACD-signal_MACD
    bar_values=bars.values
    bar_index_number=np.arange(0,len(bar_values))
    with plt.style.context('ggplot'):
        import matplotlib
        font = { 'weight' : 'bold',
            'size'   : 14}
        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(20,20))
        ax1=fig.add_subplot(311, ylabel='Stock price $')
        data2['Adj Close'].plot(ax=ax1,color='r',lw=3,label='Close price',legend=True)
        ewm_short.plot(ax=ax1,color='b',lw=2,label=str(short_period)+'-exp moving average',legend=True)
        ewm_long.plot(ax=ax1,color='m',lw=2,label=str(long_period)+'-exp moving average',legend=True)
        ax2=fig.add_subplot(312, ylabel='MACD')
        MACD.plot(ax=ax2,color='k',lw=2,label='MACD',legend=True)
        signal_MACD.plot(ax=ax2,color='r',lw=2,label='Signal',legend=True)
        ax3=fig.add_subplot(313, ylabel='MACD bars')
        x_axis = ax3.axes.get_xaxis()
        x_axis.set_visible(False)
        bars.plot(ax=ax3,color='r', kind='bar',label='MACD minus signal', legend=True,use_index=False)
        plt.savefig('MACD_spy.png')
        plt.show()
        st.pyplot(fig)
    
    #get buy and sell signals
    dfdum = pd.DataFrame()
    st.text('MACD Buy & Sell Indicators')
    df2['Signal']= np.where(df2['MACD'] > df2['signal_macd'], 1, 0)
    df2['Position'] = df2['Signal'].diff()
    df2['Buy']= np.where(df2['Position'] ==1, df2['Close'], np.NAN)
    df2['Sell']= np.where(df2['Position'] ==-1, df2['Close'], np.NAN)
    plt.figure(figsize=(30,15))
    plt.title('Close Price w/ Buy and sell signals', fontsize=18)
    plt.plot(df2['Close'], alpha = 0.5, label = 'Close')
    plt.plot(df2['signal_macd'], alpha = 0.5, label = 'EMA200')
    plt.plot(df2['MACD'], alpha = 0.5, label = 'MACD')
    plt.scatter(df2.index, df2['Buy'], alpha =1, label= 'Buy signal', marker= '^', color= 'green')
    plt.scatter(df2.index, df2['Sell'], alpha =1, label= 'Sell signal', marker= 'v', color= 'red')
    plt.xlabel('Date', fontsize =18)
    plt.ylabel('Close Price', fontsize =18)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    #dheeraj ema start
    #get buy and sell signals
    df1['Signal']= np.where(df1['EMA50'] > df1['EMA200'], 1, 0)
    df1['Position'] = df1['Signal'].diff()
    df1['Buy']= np.where(df1['Position'] ==1, df1['Close'], np.NAN)
    df1['Sell']= np.where(df1['Position'] ==-1, df1['Close'], np.NAN)
    
    st.text('EMA Buy & Sell Indicator')
    plt.figure(figsize=(30,15))
    plt.title('Close Price w/ Buy and sell signals', fontsize=18)
    plt.plot(df1['Close'], alpha = 0.5, label = 'Close')
    plt.plot(df1['EMA200'], alpha = 0.5, label = 'EMA200')
    plt.plot(df1['EMA50'], alpha = 0.5, label = 'EMA50')
    plt.scatter(df1.index, df1['Buy'], alpha =1, label= 'Buy signal', marker= '^', color= 'green')
    plt.scatter(df1.index, df1['Sell'], alpha =1, label= 'Sell signal', marker= 'v', color= 'red')
    plt.xlabel('Date', fontsize =18)
    plt.ylabel('Close Price', fontsize =18)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    #dheeraj ema end
    
    
    #dheeraj Super trend start



    atr_period = 10
    multiplier = 3
    
    high = df5['High']
    low = df5['Low']
    close = df5['Close']
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    tr = pd.concat(price_diffs, axis=1)
    tr = tr.abs().max(axis=1)
    # default ATR calculation in supertrend indicator using ewm
    atr = tr.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    
    def Supertrend(df5, atr_period, multiplier):
        
        high = df5['High']
        low = df5['Low']
        close = df5['Close']
        
        # calculate ATR
        price_diffs = [high - low, 
                       high - close.shift(), 
                       close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
        # df['atr'] = df['tr'].rolling(atr_period).mean()
        
        # HL2 is simply the average of high and low prices
        hl2 = (high + low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (multiplier * atr)
        final_lowerband = lowerband = hl2 - (multiplier * atr)
        
        # initialize Supertrend column to True
        supertrend = [True] * len(df5)
        
        for i in range(1, len(df5.index)):
            curr, prev = i, i-1
            
            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]
                
                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]
    
            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan
        
        return pd.DataFrame({
            'Supertrend': supertrend,
            'Final Lowerband': final_lowerband,
            'Final Upperband': final_upperband
        }, index=df5.index)
        
        
    atr_period = 10
    atr_multiplier = 3.0
    
    supertrend = Supertrend(df5, atr_period, atr_multiplier)
    df5 = df5.join(supertrend)
    
    
    # visualization
    st.text('SuperTrend')
    plt.figure(figsize=(30,15))
    plt.title('Close Price w/ Buy and sell signals', fontsize=18)
    plt.plot(df5['Close'], alpha = 0.5, label = 'Close')
    plt.plot(df5['Final Lowerband'], alpha = 0.5, label = 'FInal Lowerband')
    plt.plot(df5['Final Upperband'], alpha = 0.5, label = 'Final Upperband')
    plt.xlabel('Date', fontsize =18)
    plt.ylabel('Close Price', fontsize =18)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    df5
    
    
    is_uptrend = df5['Supertrend']
    close = df5['Close']
    
    # initial condition
    in_position = False
    entry = []
    exit = []
    df3 = pd.DataFrame(index=range(len(df5)),columns=range(5))
    df3.columns = ["Buy date", "Buy Price", "Sell date", "Sell price","MTM"]
    df4 = pd.DataFrame(index=range(len(df5)),columns=range(5))
    df4.columns = ["Buy date", "Buy Price", "Sell date", "Sell price","MTM"]
    for i in range(2, len(df5)):
        # if not in position & price is on uptrend -> buy
        j=0
        if not in_position and is_uptrend[i]:
            entry.append((i, close[i]))
            in_position = True
            df3.iat[j,0]= df5.index[i].strftime("%Y/%m/%d")
            df3.iat[j,1]= round(close[i],2)
        # if in position & price is not on uptrend -> sell
        elif in_position and not is_uptrend[i]:
            exit.append((i, close[i]))
            in_position = False
            df3.iat[j,2]= df5.index[i].strftime("%Y/%m/%d")
            df3.iat[j,3]= round(close[i],2)
            df3.iat[j,4]= df3.iat[j,3] - df3.iat[j,1]
            df3=df3.dropna()
            df4= df4.append(df3,ignore_index = True)
            df4=df4.dropna()
        j=j+1
    
    
    print(df4)
    wintrades=0
    totaltrades=len(df4.index)
    probability2=0
    abc = df4['MTM']
    for i in range(len(df4)):
        if(abc[i] >0):
            wintrades=wintrades+1
            probability2 = (wintrades/totaltrades)*100
    
       
    
    
    
    
    #dheeraj super trend end
    
    dfdum = df2.copy()
    dfdum = dfdum.drop(dfdum[(dfdum['Buy'].isna() & dfdum['Sell'].isna())].index)
    dfdum
    
    
    wintrades=0
    totaltrades=0
    probability=0
    investment= 100000
    shares=0
    earning=0
    roi=0
    for i in range(len(dfdum)):
        if dfdum.iloc[i]['Position']== 1:
            B=dfdum.iloc[i]['Buy']
            shares = investment/B
            i=i+1
            totaltrades=totaltrades+1
            if dfdum.iloc[i]['Position']== -1:
                S=dfdum.iloc[i]['Sell']
            profit=S-B
            earning = earning + profit
            if(profit>0):
                wintrades=wintrades+1
            probability = (wintrades/totaltrades)*100
            profit= profit * shares
            earning = earning + profit
    roi= earning/ investment
    print ("Roi X =", roi)
    if(dfdum['Buy'].iloc[-1]>0):
        Y='BUY'
    else:
        Y='SELL' 
    # DHEERAJ EMA START
    dfdum1 = df1.copy()
    dfdum1 = dfdum1.drop(dfdum1[(dfdum1['Buy'].isna() & dfdum1['Sell'].isna())].index)
    dfdum1
    wintrades1=0
    totaltrades1=0
    probability1=0
    investment1= 100000
    shares1=0
    earning1=0
    roi1=0
    for i in range(len(dfdum1)):
        if dfdum1.iloc[i]['Position']== 1:
            B1=dfdum1.iloc[i]['Buy']
            shares1 = investment1/B1
            i=i+1
            totaltrades1=totaltrades1+1
            try:
                if dfdum1.iloc[i]['Position']== -1:
                    S1=dfdum1.iloc[i]['Sell']
            except Exception as e:
                i=i+1
            profit1=S1-B1
            
            if(profit1>0):
                wintrades1=wintrades1+1
            profit1= profit1 * shares1 
            probability1 = (wintrades1/totaltrades1)*100
            earning1 = earning1 + profit1
    roi1= earning1/ investment1
    print ("Roi Y =", roi1)
    if(dfdum1['Buy'].iloc[-1]>0):
        X='BUY'
    else:
        X='SELL' 
    # DHEERAJ EMA END
    st.text('MACD Success Rate')
    st.text(Y)   
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        value = probability,
        mode = "gauge+number+delta",
        title = {'text': "Probability of Winning"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 30], 'color': "lightgray"},
                     {'range': [30, 70], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}))
    
    st.plotly_chart(fig)
    
    #DHEERAJ START EMA
    st.text('EMA Success Rate')
    st.text(X)   
    import plotly.graph_objects as go
    
    fig1 = go.Figure(go.Indicator(
        value = probability1,
        mode = "gauge+number+delta",
        title = {'text': "Probability of Winning"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 30], 'color': "lightgray"},
                     {'range': [30, 70], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}))
    
    st.plotly_chart(fig1)
    #DHEERAJ END EMA
    
    #DHEERAJ SUPER TREND
    st.text('Supertrend Success Rate')
    st.text(X)   
    import plotly.graph_objects as go
    
    fig2 = go.Figure(go.Indicator(
        value = probability2,
        mode = "gauge+number+delta",
        title = {'text': "Probability of Winning"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 30], 'color': "lightgray"},
                     {'range': [30, 70], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}))
    
    st.plotly_chart(fig2)
    #DHEERAJ super trend end
     
except:
    pass

