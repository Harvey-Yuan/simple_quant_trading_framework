import requests
import json
import config
import sys
sys.path.append('/Users/harvey/Desktop/MY_porject/quant3/config')
import path_config
import config



class alpaca_client:

    def __init__(self,alpaca_api_key = config.alpaca_api_key,alpaca_api_secret = config.alpaca_api_secret,paper=True):

        self.alpaca_api_key = alpaca_api_key
        self.alpaca_api_secret =  alpaca_api_secret
        if paper:
            self.url = 'https://paper-api.alpaca.markets/v2/'
        else:
            self.url = 'https://api.alpaca.markets/v2'

        self.header = {
            "accept": "application/json",
            "content-type": "application/json",
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_api_secret
        }

    def getOpenPosition(self):

        url = self.url+"positions"
        headers = self.header
        response = requests.get(url, headers=headers)
        return json.loads(response.text)
    
    def closeAllPostion(self):

        url = self.url+"positions"
        headers = self.header
        response = requests.delete(url, headers=headers)
        return json.loads(response.text)
    
    def closePosition(self,symbol):

        url = f"{self.url}positions/{symbol}"
        headers = self.header
        response = requests.delete(url, headers=headers)
        return json.loads(response.text)
    

    def getLastQuote(self,symbol):

        url = f"https://data.sandbox.alpaca.markets/v2/stocks/quotes/latest?symbols={symbol}&feed=sip"
        #url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?symbols={symbol}&feed=sip"

        headers = self.header

        response = requests.get(url, headers=headers)
        print(response.text)
        


    def create_limit_order(self,symbol,qty,side,limit_price,take_profit_limit_price,stop_loss_stop_price,stop_loss_limit_price,extended_hours=True):

        url = self.url+"orders"

        payload = {
            "symbol": symbol,  # 交易的股票代码
            "qty": qty,  # 购买数量
            "side": side,  # 买入还是卖出
            "type": "limit",  # 订单类型
            "time_in_force": "gtc",  # 订单的时间有效期
            "limit_price": limit_price,  # 限价买入价
            "order_class": "bracket",  # 使用Bracket订单类型以设置止盈和止损
            "take_profit": {
                "limit_price": take_profit_limit_price  # 止盈价
            },
            "stop_loss": {
                "stop_price": stop_loss_stop_price,  # 止损触发价
                "limit_price": stop_loss_limit_price  # 止损限价（可选）
            },
            "extended_hours": extended_hours  # 是否允许盘前盘后交易
        }

        headers = self.header
        response = requests.post(url, json=payload, headers=headers)
        return response.text
    
    
    def create_market_order(self,symbol,qty,side,time_in_force='day'):

        #url = "https://api.alpaca.markets/v2/orders"
        url = self.url+"orders"

        payload = {
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "symbol": symbol,
            "qty": qty
        }
        headers = self.header

        response = requests.post(url, json=payload, headers=headers)
        
        return response.text




if __name__ == "__main__":
    ac = alpaca_client()
    #openPositions = ac.getOpenPosition()
    quote_price = ac.getLastQuote('AAPL')

#ac.create_order(symbol='AAPL',qty=1,side='buy',limit_price=100,take_profit_limit_price=120,stop_loss_stop_price=80,stop_loss_limit_price=80,extended_hours=False)



#buy_detail = create_order(symbol='QQQ',qty=1,side='buy',time_in_force='day')
#create_limit_sell_order(symbol='QQQ', buy_price=445.05, qty=1, increase_percentage=0.005)