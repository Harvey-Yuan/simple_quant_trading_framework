from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import talib
from backtesting.lib import crossover,plot_heatmaps,resample_apply
import seaborn as sns
import matplotlib.pyplot as plt

print(GOOG)


talib.RSI(GOOG['Close'],14)


# define your own maximum goal
def optim_func(series):
  return series['Equity Final [$]'] / series['Exposure Time [%]']

class RsiOci(Strategy):

  upper_bound = 70
  lower_bound = 30
  rsi_window = 14

  def init(self):
    #括号里后面两个参数是传递给第一个函数的参数
    #这个self.I是内置函数，每一行都要重新计算一次
    self.daily_rsi = self.I(talib.RSI,self.data.Close,self.rsi_window)
    # self.weekly_rsi = resample_apply(
    #   "W-FRI",talib.RSI,self.data.Close,self.rsi_window
    # )
  
  def next(self):

    price = self.data.Close[-1]

    if crossover(self.daily_rsi,self.upper_bound):
        #and self.weekly_rsi[-1]>self.upper_bound):
      if self.position.is_long or not self.position:
        self.position.close()
        self.sell()
    
    elif crossover(self.lower_bound,self.daily_rsi):
      #and self.weekly_rsi[-1]<self.lower_bound):
      if self.position.is_short or not self.position:
        self.position.close()
        self.buy(tp=1.15*price,sl=0.95*price,size = 0.1)


bt = Backtest(GOOG,RsiOci,cash=10_000)

stats = bt.run()

stats = bt.optimize(upper_bound = range(50,85,5),
            lower_bound = range(10,45,5),
            rsi_window = 14,
            maximize=optim_func,
            constraint=lambda params:params.upper_bound >params.lower_bound,
            #do a random search rather than grid search
            max_tries=100
            )



print(stats)

bt.plot(filename='/Users/harvey/Desktop/MY_porject/quant3/main_prototypes/alpaca_firstProto/backTestPlot/test.html')

print(stats['_strategy'])


stats,heatmap = bt.optimize(upper_bound = range(50,85,5),
            lower_bound = range(10,45,5),
            rsi_window = 14,
            maximize=optim_func,
            constraint=lambda params:params.upper_bound >params.lower_bound,
            #do a random search rather than grid search
            max_tries=100,
            return_heatmap=True
            )

print(heatmap)

hm = heatmap.groupby(['upper_bound','lower_bound']).mean().unstack()

sns.heatmap(hm,cmap='viridis')

plot_heatmaps(heatmap)