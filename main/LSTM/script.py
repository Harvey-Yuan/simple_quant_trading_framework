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
#import config
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

main_path = os.getenv('MAIN_PATH')
first_path = os.getenv('FIRST_PATH')

state_path = os.path.join(main_path,'LSTM/state/state.pkl')
log_path = os.path.join(main_path,'LSTM/log')
model_path = [os.path.join(first_path,'strategy_develop/LSTM/202409/20240812_model.h5'),os.path.join(first_path,'strategy_develop/LSTM/202409/20240812_model.pkl')]



class LSTM_tradingBot(tradingBot):
    def __init__(self,model_path = model_path):
        super().__init__(initial_capital=50000,state_path = state_path,log_path = log_path)
        
        
        
        self.create_order = self.alpaca_client.create_market_order
        self.get_quote = self.tradier_client.get_quote
        self.close_position = self.alpaca_client.closePosition
        self.get_data = self.data_client
        self.model_path = model_path[0]
        self.component_path = model_path[1]





    def execute_trading_strategy(self,predict_change_rate, filename=os.path.join(current_dir,'state/trading_state.pkl')):
        state = self.state

        # 从状态中读取值
        capital = state['capital']
        position = state['position']
        waiting_period = state['waiting_period']
        trade_count = state['trade_count']
        
        waiting_period_set = 1
        trade_signals = 0
        buy_threshold = 0.005
        sell_threshold = -0.005
        action = ''


        if waiting_period > 0:
            waiting_period -= 1
        if waiting_period == 0 and position > 0 and predict_change_rate <= buy_threshold:
            #平仓
            current_close_price = self.get_quote('TQQQ')['last'][0]
            capital += abs(position) * current_close_price
            self.close_position('TQQQ')
            action = action.join(f'清仓TQQQ;清仓价格{current_close_price}')
            position = 0

        if waiting_period == 0 and position < 0 and predict_change_rate >= sell_threshold:

            current_close_price = self.get_quote('SQQQ')['last'][0]
            capital += abs(position) * current_close_price
            self.close_position('SQQQ')
            action = action.join(f'清仓SQQQ;清仓价格{current_close_price}')
            position = 0


        if predict_change_rate > buy_threshold and position <= 0:
            
            trade_signals = 1
            capital, position,current_close_price = buy_logic(capital, position)
            action = action.join(f'买入TQQQ价值{capital},当前每股价格为{current_close_price};')
            waiting_period = waiting_period_set
            trade_count += 1
        elif predict_change_rate < sell_threshold and position >= 0:
            
            trade_signals = -1
            capital, position,current_close_price = sell_logic(capital, position)
            action = action.join(f'买入SQQQ价值{capital},当前每股价格为{current_close_price};')
            waiting_period = waiting_period_set
            trade_count += 1
        else:
            trade_signals = 0

        total_asset_value = capital
        state['total_asset_list'].append(total_asset_value)
        state['capital_list'].append(capital)

        # 更新状态
        state.update({'capital': capital, 'position': position, 'waiting_period': waiting_period, 'trade_count': trade_count})
        self.save_state(state)
        log_info = {
            'log':action
        }
        self.save_log(log_info)

        return state




    def buy_logic(self,capital, position):
        print('执行买入逻辑')
        if position < 0:  # 如果当前持有空头仓位，先平仓
            try:
                # 平掉空头仓位
                self.close_position('SQQQ')
                print(f"平掉空头仓位: {abs(position)} 股")
                position = 0
            except Exception as e:
                print(f"平仓失败: {e}")

        # 计算可以购买的股数
        current_close_price = self.get_quote('TQQQ')['last'][0]
        shares_to_buy = float(capital / current_close_price)
        if shares_to_buy > 0 and position == 0:
            try:
                response = self.create_order('TQQQ',shares_to_buy,'buy',time_in_force='day')
                capital -= shares_to_buy * current_close_price
                position += shares_to_buy
                print(f"买入 {shares_to_buy} 股")
            except Exception as e:
                print(f"买入失败: {e}")
        return capital, position,current_close_price

    def sell_logic(self,capital, position):
        print('执行卖出逻辑')
        if position > 0:  # 如果当前持有多头仓位，先平仓
            try:
                self.close_position('TQQQ')
                print(f"平掉多头仓位: {position} 股")
                position = 0
            except Exception as e:
                print(f"平仓失败: {e}")

        # 计算可以卖出的股数（假设卖出等同于开立空头仓位）
        current_close_price = self.get_quote('SQQQ')['last'][0]
        shares_to_sell = float(capital / current_close_price)
        if shares_to_sell > 0 and position == 0:
            try:
                response = self.create_order('SQQQ',shares_to_sell,'buy',time_in_force='day')
                capital -= shares_to_sell * current_close_price
                position -= shares_to_sell
                print(f"卖出（或开空） {shares_to_sell} 股")
            except Exception as e:
                print(f"卖出失败: {e}")
        return capital, position,current_close_price




    def predict(self,data,scaler_features,scaler_label,model,time_step=16):

        df = data
        features = df.drop(['close', 'open', 'high', 'low'], axis=1)
        labels = df[['close']]  # 以'close'列作为示例标签

        scaled_features = scaler_features.transform(features)

        latest_X = scaled_features[-time_step:].reshape(1, time_step, scaled_features.shape[1])


        predicted_y = model.predict(latest_X)
        predicted_y = scaler_label.inverse_transform(predicted_y)

        last_time_index = df.index[-1]
        predicted_time = last_time_index + pd.Timedelta(hours=1)

        print(f"预测的y值对应的时间点是: {predicted_time}")
        print(f"预测的y值是: {predicted_y}")
        
        return float(predicted_y.squeeze())


# 例如，使用APScheduler定时运行


    def job_function(self):
                # 加载模型
        model = load_model(self.model_path)
        components_dict = joblib.load(self.component_path)

        scaler_features = components_dict['scaler_features']
        scaler_label = components_dict['scaler_label']
        
        #download data
        df = self.get_data("QQQ")
        lastest_date = df.index[-1]
        lastest_price = df.iloc[-1,:]['close']
        
        # 验证是否为最新数据
        ny_tz = pytz.timezone('America/New_York')
        ny_time = datetime.now(ny_tz)
        other_time = lastest_date
        hours_difference = (ny_time - other_time).total_seconds() / 3600
        if hours_difference > 0.4:
            self.log_state(None)
            #return '数据不够新'
        else:
            pass
        
        

        predict_price = self.predict(df,scaler_features = scaler_features,scaler_label = scaler_label,model = model,time_step=16)
        
        # 获取最新的predict_change_rate和current_close_price
        # 这里需要您根据实际情况来获取这些值
        predict_change_rate = predict_price/lastest_price - 1
        print('predict_change_rate',predict_change_rate)
        current_close_price = lastest_price
        state = self.execute_trading_strategy(predict_change_rate)
        self.log_state(state)
        return state



class LSTM_Strategy_Backtest(Strategy):
    model_path = [os.path.join('path/to/model', '20240812_model.h5'), os.path.join('path/to/model', '20240812_model.pkl')]
    buy_threshold = 0.005
    sell_threshold = -0.005
    waiting_period_set = 1
    
    def init(self):
        # Load the model and scalers
        self.model = load_model(self.model_path[0])
        components_dict = joblib.load(self.model_path[1])
        self.scaler_features = components_dict['scaler_features']
        self.scaler_label = components_dict['scaler_label']
        
        # Get the data (assuming `self.get_data` is available)
        self.data = self.get_data("QQQ")  # Replace "QQQ" with your ticker symbol
        self.waiting_period = 0
        
        # Initialize the state
        self.position = 0
        self.capital = 50000  # Example starting capital
        self.trade_count = 0

    def next(self):
        # Predict the next price using the model
        predict_price = self.predict(self.data.iloc[-16:], self.scaler_features, self.scaler_label, self.model)
        current_close_price = self.data['close'][-1]
        predict_change_rate = predict_price / current_close_price - 1

        # 使用提取的交易逻辑函数
        self.execute_trading_logic(predict_change_rate, current_close_price)

    def execute_trading_logic(self, predict_change_rate, current_close_price):
        # Trading logic
        action = ''
        if self.waiting_period > 0:
            self.waiting_period -= 1

        if self.waiting_period == 0 and self.position > 0 and predict_change_rate <= self.buy_threshold:
            self.sell(size=abs(self.position))
            self.position = 0
            action = f'清仓TQQQ;清仓价格{current_close_price}'

        elif self.waiting_period == 0 and self.position < 0 and predict_change_rate >= self.sell_threshold:
            self.buy(size=abs(self.position))
            self.position = 0
            action = f'清仓SQQQ;清仓价格{current_close_price}'

        elif predict_change_rate > self.buy_threshold and self.position <= 0:
            self.buy(size=0.1 * self.capital / current_close_price)
            self.position += 0.1 * self.capital / current_close_price
            self.waiting_period = self.waiting_period_set
            self.trade_count += 1
            action = f'买入TQQQ价值{0.1 * self.capital},当前每股价格为{current_close_price};'

        elif predict_change_rate < self.sell_threshold and self.position >= 0:
            self.sell(size=0.1 * self.capital / current_close_price)
            self.position -= 0.1 * self.capital / current_close_price
            self.waiting_period = self.waiting_period_set
            self.trade_count += 1
            action = f'买入SQQQ价值{0.1 * self.capital},当前每股价格为{current_close_price};'

        # Log the action
        self.save_log({'log': action})

    def predict(self, data, scaler_features, scaler_label, model, time_step=16):
        features = data.drop(['close', 'open', 'high', 'low'], axis=1)
        scaled_features = scaler_features.transform(features)

        latest_X = scaled_features[-time_step:].reshape(1, time_step, scaled_features.shape[1])
        predicted_y = model.predict(latest_X)
        predicted_y = scaler_label.inverse_transform(predicted_y)

        return float(predicted_y.squeeze())

    def save_log(self, log_info):
        # Implement log saving
        pass

    def get_data(self, ticker):
        # Mock implementation; replace with actual data retrieval logic
        return pd.DataFrame({'close': np.random.random(500), 'open': np.random.random(500), 'high': np.random.random(500), 'low': np.random.random(500)})



if __name__ == "__main__":
    bot = LSTM_tradingBot()
    state = bot.job_function()
    print(state)



    # Example of running a backtest with this strategy
    bt = Backtest(data=None,  # Replace `None` with actual data
                strategy=LSTM_Strategy,
                cash=50000,
                exclusive_orders=True)

    # You can run and optimize the strategy similar to the previous example
    stats = bt.run()
    print(stats)
    bt.plot()
