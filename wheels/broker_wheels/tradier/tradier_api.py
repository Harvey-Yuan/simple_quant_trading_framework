#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:00:33 2023

@author: harvey
"""

import requests
import sys
sys.path.append('/Users/harvey/Desktop/MY_porject/quant3/config')
import path_config
import config
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd


class tradier_client:
    def __init__(self,token = config.demo_token,api_url = config.demo_api_base_url,account_id = config.demo_id):
        self.token = token
        self.api_url = api_url
        self.account_id = account_id


    #get quote
    def get_quote(self,ticker):

        headers={'Authorization': 'Bearer {}'.format(self.token), 
                'Accept': 'application/json'}
        
        quote_url = '{}markets/quotes'.format(self.api_url)
        
        response = requests.get(quote_url,
            params={'symbols': ticker},
            headers = headers)
        json_response = response.json()
        quote_dict = json_response['quotes']['quote']
        df = pd.DataFrame(quote_dict, index=[0])
        return df


    '''
    df = get_quote('AAPL')


    df['last']
    df['bid']
    df['ask']
    '''
    #df_quote = get_quote('AAPL')


    def candles(self,
                ticker,
                interval='5min',
                start = datetime.combine(datetime.now().date(), time(9, 30)).strftime('%Y-%m-%d %H:%M'),
                end = datetime.now().strftime('%Y-%m-%d %H:%M')):
        
        headers={'Authorization': 'Bearer {}'.format(self.token), 
                'Accept': 'application/json'}
        url = 'https://api.tradier.com/v1/markets/timesales'

        params={'symbol': ticker, 'interval': interval, 'start': start, 'end': end, 'session_filter': 'all'}

        response = requests.get(url,
            params=params,
            headers=headers
        )
        json_response = response.json()

        data_list = json_response['series']['data']

        df = pd.DataFrame(data_list)
        return df

    '''
    df = candles('AAPL')



    df['time']
    df['timestamp']
    df['open']
    df['high']
    df['low']
    df['close']
    df['volume']
    '''
    #df_candle = candles('AAPL')

    #equity order
    def equity_order(self,ticker,side = 'buy',quantity = 1,trade_type = 'market',price = 1,stop = 1):

        headers={'Authorization': 'Bearer {}'.format(self.token), 
                'Accept': 'application/json'}
        
        data = {'class': 'equity', 'symbol': ticker, 'side': side, 'quantity': quantity, 'type': trade_type,'duration': 'day'
            #, , 'price': '1.00', 'stop': '1.00', 'tag': 'my-tag-example-1'
            }
        
        url = '{}accounts/{}/orders'.format(self.api_url,self.account_id)

        response = requests.post(url,
            data=data,
            headers = headers
            
        )
        json_response = response.json()
        df = pd.DataFrame(json_response['order'],index=[0])
        return df
    '''
    equity_order('AAPL')
    '''



if __name__ =='__main__':
    tc = tradier_client()
    quote = tc.get_quote('AAPL')