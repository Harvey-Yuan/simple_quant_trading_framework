from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd  # 假设您已经有了pandas库
from keras.models import load_model
from datetime import datetime
import pytz
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import requests
import pickle
import logging
from datetime import datetime
import os
import config
from backtesting import Backtest, Strategy

#config cuurent path
current_dir = os.path.dirname(os.path.abspath(__file__))

#config path
import sys
sys.path.append('/Users/harvey/Desktop/MY_porject/quant3/config')
import path_config
from alpaca_firstProto.scriptProto import tradingBot
from data_wheels.yfinance.factor_adding import get_data

main_path = os.getenv('MAIN_PATH')
first_path = os.getenv('FIRST_PATH')

state_path = os.path.join(main_path,'LSTM/state/state.pkl')
log_path = os.path.join(main_path,'LSTM/log')
model_path = [os.path.join(first_path,'strategy_develop/LSTM/202409/20240816_model.h5'),os.path.join(first_path,'strategy_develop/LSTM/202409/20240816_model.pkl')]


class LSTM_Strategy(tradingBot):
    def __init__(self, model_path=model_path,initial_capital=50000,state_path = state_path,log_path=log_path):

        self.model = load_model(model_path[0])
        components_dict = joblib.load(model_path[1])
        self.scaler_features = components_dict['scaler_features']
        self.scaler_label = components_dict['scaler_label']
        self.dataframe = components_dict['dataframe']



        #绑定数据获取
        self.data_client = get_data


        #绑定log
        self.log_path = log_path

        #策略参数
        self.initial_capital = initial_capital
        self.buy_threshold = 0.005
        self.sell_threshold = -0.005
        self.waiting_period_set = 1


        #绑定自己的state path
        self.state_path = state_path
        self.load_state()
        if not self.state:
            self.waiting_period = 1
            self.position = 0
            self.upload_state()
            
        self.waiting_period = self.state['waiting_period']
        self.position = self.state['position']


    def upload_state(self):
        self.state = {
                'waiting_period':self.waiting_period_set,
                'position':self.position
            }
        self.save_state()


    def predict(self, data, time_step=16):
        '''
        input:任意大于16 step的数据
        output：一个预测的y值
        '''
        features = data.drop(['Close', 'Open', 'High', 'Low'], axis=1)
        scaled_features = self.scaler_features.transform(features)
        latest_X = scaled_features[-time_step:].reshape(1, time_step, scaled_features.shape[1])
        predicted_y = self.model.predict(latest_X)
        predicted_y = self.scaler_label.inverse_transform(predicted_y)
        return float(predicted_y.squeeze())
    
    def get_latest_df(self,ticker='QQQ'):
        return self.data_client(ticker)
        

    def get_signal(self, backtest=False, backtest_index=100):
        '''
        output:决定最新数据的买卖信号
        '''
        if not backtest:
            data = self.get_latest_df('QQQ')
            last_close_price = data['Close'].iloc[-1]
            predict_price = self.predict(data)
        else:
            data = self.dataframe.iloc[:backtest_index]
            last_close_price = data['Close'].iloc[-1]
            predict_price = data['predicted_close_in1h'].iloc[-1]

        
        predict_change_rate = predict_price / last_close_price - 1


        if self.waiting_period > 0:
            self.waiting_period -= 1
            self.upload_state()

            return 'hold'  # 保持不动
        
        action = 'hold'
    
        
        #买入信号消失+有买入仓位+等待时间到=卖出清仓
        if self.waiting_period == 0 and self.position > 0 and predict_change_rate <= self.buy_threshold:
            action = 'close'
            self.position = 0
        

        #卖出信号消失+有卖出仓位+等待时间到=买入清仓
        elif self.waiting_period == 0 and self.position < 0 and predict_change_rate >= self.sell_threshold:
            action = 'close'
            self.position = 0
        

        #买入建仓
        elif predict_change_rate > self.buy_threshold and self.position <= 0:
            action = 'closeNbuy'
            self.position = 1
    
            self.waiting_period = self.waiting_period_set


        #卖出建仓
        elif predict_change_rate < self.sell_threshold and self.position >= 0:
            action = 'closeNsell'
            self.position = -1
            self.waiting_period = self.waiting_period_set


        self.upload_state()

        return action


class LSTM_Strategy_Backtest(Strategy):
    def init(self):
        self.current_index = 0
    

        
    def next(self):

        signal = self.data.df['signal'].iloc[self.current_index]

        # 根据信号进行交易
        if signal == 'close':
            self.position.close()

        elif signal == 'closeNbuy':
            self.position.close()
            self.buy(size=1)

        elif signal == 'closeNsell':
            self.position.close()
            self.sell(size=1)

        # 更新索引
        self.current_index += 1


if __name__ == '__main__':
    LSTMStrategy = LSTM_Strategy()
    df = LSTMStrategy.dataframe[-200:]
    signals=[]
    for i in range(len(df)):
        signals.append(LSTMStrategy.get_signal(backtest=True,backtest_index=i+1))
    
    df['signal'] = signals

    bt = Backtest(df,LSTM_Strategy_Backtest,cash=10_000)
    stats = bt.run()
    print(stats)
    bt.plot(filename='/Users/harvey/Desktop/MY_porject/quant3/main_prototypes/alpaca_firstProto/backTestPlot/test.html')

    #检验数据准确度
    # LSTMStrategy.dataframe.iloc[12:30].iloc[-1]['predicted_close_in1h']
    # LSTMStrategy.predict(LSTMStrategy.dataframe.iloc[12:30].drop(['actual_close_in1h','predicted_close_in1h'],axis=1))



        
class LSTM_tradingBot(tradingBot):
    def __init__(self):
        super().__init__(initial_capital=50000, state_path=state_path, log_path=log_path)

        #init strategy instance
        self.strategy = LSTM_Strategy(model_path)

        self.create_order = self.alpaca_client.create_market_order
        self.get_quote = self.tradier_client.get_quote
        self.close_position = self.alpaca_client.closePosition
        self.get_data = self.data_client


    def execute_trading_strategy(self):

        data = self.get_data("QQQ")
        signal, price = self.strategy.get_signal(data)

        if signal == 'buy':
            self.buy_logic(price)
        elif signal == 'sell':
            self.sell_logic(price)


    def buy_logic(self, price):
        current_price = self.get_quote('TQQQ')['last'][0]
        self.strategy.update_position('buy', current_price)
        self.create_order('TQQQ', self.strategy.position, 'buy', time_in_force='day')
        print(f"Bought at {current_price}")


    def sell_logic(self, price):
        current_price = self.get_quote('SQQQ')['last'][0]
        self.strategy.update_position('sell', current_price)
        self.create_order('SQQQ', self.strategy.position, 'sell', time_in_force='day')
        print(f"Sold at {current_price}")


