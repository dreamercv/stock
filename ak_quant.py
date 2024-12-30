# -*- coding: utf-8 -*-
# @File   : ak_quant.py
# @Time   : 2024/12/20 下午5:17 
# @Author : binge.van
# @decs   :


import os

import tushare as ts
def main():
    pass
from datetime import datetime

import backtrader as bt  # 升级到最新版
import matplotlib.pyplot as plt  # 由于 Backtrader 的问题，此处要求 pip install matplotlib==3.2.2
import akshare as ak  # 升级到最新版
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def get_stocks_from_tushare( biying_stock_list=[], max_price=20, threshs=[500000000, 10000000000]):  #
    """
    目前只根据市值选股票
    Args:
        tushare_licence:
        threshs:

    Returns:

    """
    tushare_licence = "8f7419cb3d0bc956544a972de149e0e4263895e9467fef74f09b21c2"
    ts.set_token(tushare_licence)
    df = ts.realtime_list(src='dc')
    stock_codes, sh_codes, sz_codes = [], [], []
    for i in range(len(df)):
        total_mv = df.iloc[i, 16]
        dm, jys = df.iloc[i, 0].split(".")
        jys = jys.lower()
        mc = df.iloc[i, 1]
        price = df.iloc[i, 3]  # 当前价格
        if "bj" in jys: continue  # 去掉北京的股票
        if len(biying_stock_list) > 0 and dm not in biying_stock_list: continue  # 去掉不在biying的股票，必应中没有北交所的股票
        if "688" in dm or "300" in dm or "301" in dm: continue  # 跳过创业板等不能买的股票
        if price > max_price * 1.1: continue  # 过滤本金不足的股票
        if total_mv < threshs[0] and total_mv > threshs[1]: continue  # 选择市值在5亿到100亿的股票

        stock_codes.append({
            "dm": dm,
            "mc": mc,
            "jys": jys
        })

        # 分上证指数和深圳指数
        # 待完善
    return stock_codes

class MyStrategy(bt.Strategy):
    """
    主策略程序
    """
    params = (("maperiod", 20),)  # 全局设定交易策略的参数

    def __init__(self):
        """
        初始化函数
        """
        self.data_close = self.datas[0].close  # 指定价格序列
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        # 添加移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )

    def next(self):
        """
        执行逻辑
        """
        if self.order:  # 检查是否有指令等待执行,
            return
        # 检查是否持仓
        if not self.position:  # 没有持仓
            if self.data_close[0] > self.sma[0]:  # 执行买入条件判断：收盘价格上涨突破20日均线
                self.order = self.buy(size=100)  # 执行买入
        else:
            if self.data_close[0] < self.sma[0]:  # 执行卖出条件判断：收盘价格跌破20日均线
                self.order = self.sell(size=100)  # 执行卖出


def test(code):
    # 利用 AKShare 获取股票的后复权数据，这里只获取前 7 列
    stock_hfq_df = ak.stock_zh_a_hist(symbol=code, adjust="hfq").iloc[:, :7]
    # 删除 `股票代码` 列
    del stock_hfq_df['股票代码']
    # 处理字段命名，以符合 Backtrader 的要求
    stock_hfq_df.columns = [
        'date',
        'open',
        'close',
        'high',
        'low',
        'volume',
    ]
    # 把 date 作为日期索引，以符合 Backtrader 的要求
    stock_hfq_df.index = pd.to_datetime(stock_hfq_df['date'])




    cerebro = bt.Cerebro()  # 初始化回测系统
    start_date = datetime(2024, 4, 1)  # 回测开始时间
    end_date = datetime(2024,12,19)  # 回测结束时间
    data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
    cerebro.adddata(data)  # 将数据传入回测系统
    cerebro.addstrategy(MyStrategy)  # 将交易策略加载到回测系统中
    start_cash = 1000000
    cerebro.broker.setcash(start_cash)  # 设置初始资本为 100000
    cerebro.broker.setcommission(commission=0.002)  # 设置交易手续费为 0.2%
    cerebro.run()  # 运行回测系统

    port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    pnl = port_value - start_cash  # 盈亏统计

    print(f"{code}: {start_cash}")
    print(f"总资金: {round(port_value, 2)}")
    print(f"净收益: {round(pnl, 2)}, {round(pnl/start_cash, 6)*100}")
    print()

    return round(pnl/start_cash, 2)*100

if __name__ == '__main__':
    stocks = get_stocks_from_tushare()
    p0,p3,p5,l0 = [],[],[],[]
    for stock in stocks:
        code = stock["dm"]
        pct = test(code)
        if pct < 0:
            l0.append(code)
        elif pct < 30:
            p0.append(code)
        elif pct < 50:
            p3.append(code)
        else:
            p5.append(code)

    final_ = np.array([len(l0),len(p0),len(p3),len(p5)])
    print(final_,fianl / sum(final_))