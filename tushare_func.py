# -*- coding: utf-8 -*-
# @File   : tushare_func.py
# @Time   : 2024/12/4 上午9:38 
# @Author : binge.van
# @decs   :


import os
import time

import tushare as ts
import json
from config import config


def one_stock_updown_amount(config):
    buy_infos = config["buy_infos"]
    sell_infos = config["sell_infos"]
    hold_infos = config["hold_infos"]
    nostock_updown = config["updown"]
    expected_funds = config["expected_funds"]



    # 操作前持仓总金额
    hold_amount_before_operation = -1*nostock_updown #操作前持仓总金额
    for stock, infos in hold_infos.items():
        hold_price, hold_num, updown_amount = infos
        hold_amount_before_operation += hold_price * hold_num
        if updown_amount < 0:
            hold_amount_before_operation -= updown_amount

    hold_updown_amount = 0 # 持仓总亏损
    sell_amount = 0  # 卖出资金
    # 跟新卖出
    sellstocks = sell_infos.keys()
    for stock in sellstocks:
        sell_price, sell_num = sell_infos[stock]
        hold_price, hold_num, updown_amount_pre = hold_infos[stock]
        remain_num = hold_num - sell_num
        updown_amount = (sell_price - hold_price) * sell_num
        sell_amount += sell_price * sell_num
        # 当前总盈亏
        cur_stock_undown = updown_amount + updown_amount_pre



        # 在持有股票中记录在持有股票信息中，否则单独记录
        if remain_num > 0:  # 卖一部分，记录一下剩余的数量，和卖掉部分的盈亏 ,盈利不计算
            if cur_stock_undown < 0:  # 当亏损是，需要将该部分资金记录到总资金中
                hold_updown_amount += cur_stock_undown
            hold_infos[stock] = [hold_price, remain_num, cur_stock_undown if cur_stock_undown < 0 else 0]  #
        else:  # 全卖 ，删除该股票记录
            hold_infos.pop(stock)
            nostock_updown += cur_stock_undown if cur_stock_undown < 0 else 0

    hold_total_amount = sell_amount # 计算持有的股票总价值
    for stock, infos in hold_infos.items():
        hold_price, hold_num, updown_amount = infos
        hold_total_amount += hold_price * hold_num  # 计算持有的股票总价值

    total_updown = hold_updown_amount + nostock_updown
    hold_total_amount -= total_updown


    # 跟新买入
    buystocks = buy_infos.keys()
    for stock in buystocks:
        if stock in hold_infos.keys():
            pre_price, pre_num, updown = hold_infos[stock]
            cur_price, cur_num = buy_infos[stock]
            total_num = pre_num + cur_num
            avg_price = (pre_price * pre_num + cur_price * cur_num) / total_num
            hold_infos[stock] = [avg_price, total_num, updown]
        else:
            hold_infos[stock] = [buy_infos[stock][0], buy_infos[stock][1], 0]



    config["hold_infos"] = hold_infos # 跟新持有
    config["buy_infos"] = {}#重置买入
    config["sell_infos"] = {} # 重置卖出
    config["updown"] = nostock_updown  # 重置卖出



    print(config)


def realtime(config):
    # with open("config.json", 'r', encoding='utf-8') as f:
    #     config = json.load(f)
    #
    # one_stock_updown_amount(config)

    # 设置你的token，登录tushare在个人用户中心里拷贝
    ts.set_token(config["token"])
    # df1 = ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')
    # by_amount = sum([code_infos[1] * code_infos[2] for code_infos in config["buy_infos"]])  # 买入资金
    # sell_amount = sum([code_infos[1] * code_infos[2] for code_infos in config["sell_infos"]])  # 卖出资金
    # updown_amount =
    while True:

        # sina数据
        total_profit = 0
        total_cur_amount = 0
        print()
        for i in range(len(config["codes_price"])):
            in_price = config["codes_price"][i][1]
            code = config["codes_price"][i][0]
            num = config["codes_price"][i][2]

            df = ts.realtime_quote(ts_code=code, src='dc')
            # df_tick = ts.realtime_tick(ts_code=code, src='dc')
            values = list(df.iloc[0])
            current_price = values[6]
            nowtime = values[3]
            name = values[0]
            pct = round((current_price - in_price) / current_price * 100, 2)
            profit = round(in_price * num * pct / 100, 2)
            total_profit += profit

            total_cur_amount += current_price * num

            print_message = [f"{str(pct)}%", profit, current_price, in_price, nowtime, code, name]
            # titles = "pct, price, " + ", ".join(list(map(str, titles)))
            values = f", ".join(list(map(str, print_message)))
            print(values)

            # print(values)
        # total_profit = total_cur_amount - in_amount
        # pct_total_profit = total_profit / total_cur_amount
        # print(str(round(pct_total_profit * 100, 2)) + "%", round(total_profit, 2), round(in_amount, 2))
        time.sleep(1 * 20)


if __name__ == '__main__':
    realtime(config)
