# Set path
import sys
sys.path.append('/Users/harvey/Desktop/MY_porject/quant3/config')
import path_config
import config
# Import necessary wheels


import os
import pickle
import requests
import json
from datetime import datetime, time, timedelta

# Trading client
from broker_wheels.alpaca.alpaca_api import alpaca_client
from broker_wheels.tradier.tradier_api import tradier_client
from data_wheels.yfinance.factor_adding import get_data


class tradingBot:
    def __init__(self,state_path = '',log_path=''):
        
        
        #broker credentials
        self.alpaca_client = alpaca_client()
        self.tradier_client = tradier_client()


        # Bind own state path
        self.state_path = state_path
        self.state = self.load_state()

        # Bind log
        self.log_path = log_path
        self.setup_log()

        

    def save_state(self):
        '''
        Save current state to state path file
        '''
        directory = os.path.dirname(self.state_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.state, f)


    def load_state(self):
        '''
        Read state from state path file and update
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
        Initialize logging.
        """
        self.log_filename = os.path.join(self.log_path, datetime.now().strftime('logfile_%Y-%m-%d.log'))
        if not os.path.exists(self.log_filename):
            # If the directory where the log file is located does not exist, create the directory first
            os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
            # Create log file
            with open(self.log_filename, 'w') as f:
                f.write(f"Log file created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


    def save_log(self, message):
        """
        Record execution time and state to log file.
        """
        self.setup_log()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_filename, 'a') as f:
            f.write(f"{now} - {message}\n")
        
        


if __name__ == "__main__":
    test = tradier(state_path = '/Users/harvey/Desktop/MY_porject/quant3/main_prototypes/alpaca_firstProto/state/test.pkl')