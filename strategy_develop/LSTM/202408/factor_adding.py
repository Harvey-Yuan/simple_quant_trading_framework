
import numpy as np
import pandas as pd  # 假设您已经有了pandas库
from datetime import datetime
import pytz
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import yfinance as yf
import requests
#import config
import numpy as np
import pandas as pd
import requests
import pickle
import logging
from datetime import datetime
import os
import config

alpaca_api_key = config.alpaca_api_key
alpaca_api_secret =  config.alpaca_api_secret

def calculate_sma(df, column='close', periods=[5, 10, 20, 50, 100, 200]):
    for period in periods:
        df[f'SMA_{period}'] = df[column].rolling(window=period).mean()

def calculate_ema(df, column='close', periods=[5, 10, 20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, column='close', periods=[14, 21, 28]):
    for period in periods:
        delta = df[column].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

def calculate_macd(df, column='close', fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = fast_ema - slow_ema
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

def calculate_bollinger_bands(df, column='close', period=20):
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    df['Bollinger_Upper'] = sma + (std * 2)
    df['Bollinger_Lower'] = sma - (std * 2)


def calculate_atr(df, high_col='high', low_col='low', close_col='close', period=14):
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    df['ATR'] = atr
    return true_range  # 返回真实范围（TR）供ADX计算使用

def calculate_adx(df, high_col='high', low_col='low', close_col='close', period=14):
    plus_dm = df[high_col].diff()
    minus_dm = -df[low_col].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # 获取单期TR并用于计算
    tr = calculate_atr(df, high_col, low_col, close_col, 1)  # 此处现在会接收到TR的序列
    tr_sum = tr.rolling(window=period).sum()  # 使用TR序列计算N期滚动和

    plus_dm_sum = plus_dm.rolling(window=period).sum()
    minus_dm_sum = minus_dm.rolling(window=period).sum()

    plus_di = 100 * (plus_dm_sum / tr_sum)
    minus_di = 100 * (minus_dm_sum / tr_sum)
    
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.rolling(window=period).mean()


def calculate_vwap(df):
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['VWAP'] = vwap

def calculate_stochastic_oscillator(df, high_col='high', low_col='low', close_col='close', k_period=14, d_period=3):
    low_min = df[low_col].rolling(window=k_period).min()
    high_max = df[high_col].rolling(window=k_period).max()
    df['%K'] = (df[close_col] - low_min) * 100 / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=d_period).mean()



def calculate_momentum(df, column='close', periods=[5, 10, 15, 20, 50, 100]):
    for period in periods:
        df[f'Momentum_{period}'] = df[column].diff(period)



def get_data(ticker = 'QQQ'):
    ticker = ticker

    data = yf.download(tickers = ticker, period="2y", interval="1h",prepost=True)
    #data = yf.download(tickers = ticker, period="2y", interval="1h")
    data = data[['Open','High','Low','Close','Volume']]
    data.columns = ['open','high','low','close','volume']
    calculate_sma(data,column='close')
    calculate_ema(data,column='close')
    calculate_rsi(data,column='close')
    calculate_macd(data,column='close')
    calculate_bollinger_bands(data,column='close')
    calculate_adx(data,close_col='close')
    calculate_vwap(data)
    calculate_vwap(data)
    calculate_stochastic_oscillator(data)
    calculate_atr(data)
    calculate_momentum(data)

    data.index = pd.to_datetime(data.index)

    data['day_of_week'] = data.index.dayofweek

    days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    data['day_of_week_name'] = data['day_of_week'].apply(lambda x: days[x])

    week_dummies = pd.get_dummies(data['day_of_week_name'], prefix='day')

    data = pd.concat([data, week_dummies], axis=1)

    data.drop(['day_of_week', 'day_of_week_name'], axis=1, inplace=True)

    data['week_of_year'] = data.index.isocalendar().week

    data['hour_of_day'] = data.index.hour
    
    data.index.name = 'Date'
    
    ######################第二版
    df = data.copy()
    vix_symbol = "^VIX"

    # 使用 download 函数获取 VIX 数据
    vix_data = yf.download(vix_symbol)
    vix_data = vix_data.tz_localize('America/New_York')
    vix_data = vix_data[['Close']]
    vix_data.columns = ['VIX']
    df_info = yf.Ticker("QQQ")
    dividends = pd.DataFrame(df_info.dividends)

    gld = yf.Ticker("GLD")
    gld = gld.history(period="5y") 
    gld['capital'] = gld['Close'] * gld['Volume']
    gld = gld[['Close','Volume','capital']]
    gld = gld.add_prefix('GLD_')

    IEF = yf.Ticker("IEF")
    IEF = IEF.history(period="5y")
    IEF['capital'] = IEF['Close'] * IEF['Volume']
    IEF = IEF[['Close','Volume','capital']]
    IEF = IEF.add_prefix('IEF_')

    DXY = yf.Ticker("USD")
    DXY = DXY.history(period="5y")
    DXY['capital'] = DXY['Close'] * DXY['Volume']
    DXY = DXY[['Close','Volume','capital']]
    DXY = DXY.add_prefix('DXY_')
    USO = yf.Ticker("USO")
    USO = USO.history(period="5y")
    USO['capital'] = USO['Close'] * USO['Volume']
    USO = USO[['Close','Volume','capital']]
    USO = USO.add_prefix('USO_')
    # 将小时级别数据的索引转换为仅包含日期的格式
    original_index = df.index

    df['date'] = df.index.date


    dividends['date'] = dividends.index.date
    gld['date'] = gld.index.date
    IEF['date'] = IEF.index.date
    DXY['date'] = DXY.index.date
    USO['date'] = USO.index.date
    vix_data['date'] = vix_data.index.date

    # 使用新的日期列来合并数据，而不是直接使用索引
    # 注意：这里假设vix_data已经有了一个叫做'date'的列
    hourly_data_merged = df.merge(vix_data, how='left', on='date')
    hourly_data_merged['VIX'] = hourly_data_merged['VIX'].fillna(0)
    hourly_data_merged = hourly_data_merged.merge(dividends, how='left', on='date')
    hourly_data_merged['Dividends'] = hourly_data_merged['Dividends'].fillna(0)

    hourly_data_merged = hourly_data_merged.merge(gld, how='left', on='date')
    hourly_data_merged[gld.columns] = hourly_data_merged[gld.columns].fillna(0)

    hourly_data_merged = hourly_data_merged.merge(IEF, how='left', on='date')
    hourly_data_merged[IEF.columns] = hourly_data_merged[IEF.columns].fillna(0)

    hourly_data_merged = hourly_data_merged.merge(DXY, how='left', on='date')
    hourly_data_merged[DXY.columns] = hourly_data_merged[DXY.columns].fillna(0)

    hourly_data_merged = hourly_data_merged.merge(USO, how='left', on='date')
    hourly_data_merged[USO.columns] = hourly_data_merged[USO.columns].fillna(0)




    hourly_data_merged.index = original_index

    # 如果不再需要date列，可以选择将其从合并后的数据框中移除
    hourly_data_merged.drop(columns=['date'], inplace=True)


    # 定义ETF列表
    etfs = ["SPY", "DIA", "IWM", "VTI"]

    # 初始化结果字典
    results = {}

    # 遍历ETF列表，下载数据并计算
    for etf in etfs:
        data = yf.download(etf, period="2y", interval="1h",prepost=True)
        data = data[['Close','Volume','High','Low']]
        data.columns = ['close','volume','high','low']
        window_short = 2  # 短期窗口
        window_long = 10  # 长期窗口
        data['capital'] = data['close']*data['volume']
        data['capital_short'] = data['capital'].rolling(window=window_short).mean()
        data['capital_long'] = data['capital'].rolling(window=window_long).mean()
        data['Volume_Change'] = data['capital_short'] - data['capital_long']
        data['Volume_Change_percent'] = data['capital_short']/data['capital_long']-1
        calculate_sma(data,periods=[5,10,20])
        calculate_sma(data,periods=[5,10,20])
        calculate_bollinger_bands(data, column='close', period=20)
        calculate_rsi(data)
        calculate_macd(data)
        calculate_atr(data)
        calculate_adx(data)
        calculate_momentum(data, periods=[5, 10, 15])
        data = data.add_prefix(etf+'_')
        results[etf] = data

        hourly_data_merged = hourly_data_merged.merge(data, how='left', left_index=True, right_index=True)
        hourly_data_merged[data.columns] = hourly_data_merged[data.columns].fillna(method = 'ffill')


    

    return hourly_data_merged
