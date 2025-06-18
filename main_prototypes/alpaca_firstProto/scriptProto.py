
#设置路径
import sys
sys.path.append('/Users/harvey/Desktop/MY_porject/quant3/config')
import path_config
import config
#导入必要的wheels


import os
import pickle
import requests
import json
from datetime import datetime, time, timedelta

#交易client
from broker_wheels.alpaca.alpaca_api import alpaca_client
from broker_wheels.tradier.tradier_api import tradier_client
from data_wheels.yfinance.factor_adding import get_data


class tradingBot:
    def __init__(self,state_path = '',log_path=''):
        
        
        #broker credentials
        self.alpaca_client = alpaca_client()
        self.tradier_client = tradier_client()


        #绑定自己的state path
        self.state_path = state_path
        self.state = self.load_state()

        #绑定log
        self.log_path = log_path
        self.setup_log()

        

    def save_state(self):
        '''
        把当前的state保存到state path文件下
        '''
        directory = os.path.dirname(self.state_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.state, f)


    def load_state(self):
        '''
        读取state path文件下的state并更新
        '''
        try:
            with open(self.state_path, 'rb') as f:
                self.state = pickle.load(f)
                return self.state
        except FileNotFoundError:
            self.state = None
            return None
        
    
    def setup_log(self):
        """
        初始化日志。
        """
        self.log_filename = os.path.join(self.log_path, datetime.now().strftime('logfile_%Y-%m-%d.log'))
        if not os.path.exists(self.log_filename):
            # 如果日志文件所在的目录不存在，则先创建目录
            os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
            # 创建日志文件
            with open(self.log_filename, 'w') as f:
                f.write(f"Log file created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


    def save_log(self, message):
        """
        记录执行时间和状态到日志文件。
        """
        self.setup_log()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_filename, 'a') as f:
            f.write(f"{now} - {message}\n")
        
        


if __name__ == "__main__":
    test = tradier(state_path = '/Users/harvey/Desktop/MY_porject/quant3/main_prototypes/alpaca_firstProto/state/test.pkl')