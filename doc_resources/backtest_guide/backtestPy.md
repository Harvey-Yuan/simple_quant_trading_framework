本文档介绍 backtest.py 的用法

backtest 接收这样的数据结构

![alt text](image.png)

会在前一天的 close 计算 indicator，在第二天的 open 价格买卖。因此这两列必需。另 index 需要是日期格式。
