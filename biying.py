# -*- coding: utf-8 -*-
# @File   : biying.py
# @Time   : 2024/12/12 下午2:28 
# @Author : binge.van
# @decs   :

"doc : http://ad.biyingapi.com/zdgc.html"
import os
import time

import requests
import tushare as ts
from datetime import datetime, date
import logging
def get_logger(filename, verbosity=1, name=None,mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

from tushare_func import realtime


# http://api.biyingapi.com/hslt/list/42FC31E3-6DE7-43A8-B596-B865257A1025
# http://api.biyingapi.com/hslt/ztgc/2024-12-11/42FC31E3-6DE7-43A8-B596-B865257A1025
# http://api.biyingapi.com/hszb/fsjy/603309/15m/42FC31E3-6DE7-43A8-B596-B865257A1025
# http://api.biyingapi.com/hszbl/fsjy/603309/15m/42FC31E3-6DE7-43A8-B596-B865257A1025


def get_requests(url):
    # 发送HTTP GET请求
    response = requests.get(url)

    # 确保请求成功
    if response.status_code == 200:
        data = response.json()
    else:
        data = None

    return data


def get_stocks_from_biying(list_url, biying_licence):
    stock_list = []
    stock_info = get_requests(f"{list_url.rstrip('/')}/{biying_licence}")
    if stock_info is not None:
        for info in stock_info:
            stock_list.append(info["dm"])
    return stock_list


def get_stocks_from_tushare(tushare_licence, biying_stock_list, max_price=20, threshs=[500000000, 10000000000]):  #
    """
    目前只根据市值选股票
    Args:
        tushare_licence:
        threshs:

    Returns:

    """
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
    return stock_codes, sh_codes, sz_codes


def save_info_biying(cur_date, realtime_info):
    info = open(os.path.join(cur_date, f"{stock_code}.txt"), "a+")
    # 将实时数据进行保存
    values = []
    for s in total_list:
        try:
            values.append(str(realtime_info[s]))
        except:
            print(11)
    values_str = ",".join(values)
    info.write(values_str + "\n")
    info.close()
    print(stock_code, values_str)


def save_info_tushare(stock_code, cur_date, realtime_info):
    info = open(os.path.join(cur_date, f"{stock_code}.txt"), "a+")
    # 将实时数据进行保存
    current_min = datetime.now().strftime('%H:%M')
    values_str = ",".join(list(map(str, realtime_info + [current_min])))
    info.write(values_str + "\n")
    info.close()
    print(current_min,stock_code, values_str)


def realtime_writing(tushare_licence,save_root
                     ):  #
    """
    目前只根据市值选股票
    Args:
        tushare_licence:
        threshs:

    Returns:

    """
    cur_date = datetime.now().strftime('%Y%m%d')
    save_path = os.path.join(save_root,"tushare", cur_date)
    os.makedirs(save_path, exist_ok=True)
    ts.set_token(tushare_licence)
    df = ts.realtime_list(src='dc')


    for i in range(len(df)):
        data = df.iloc[i].tolist()

        code = data[0]
        save_info_tushare(code, save_path, data)



def choose_stock(tushare_licence,
                 jiaoyisuo="sh",
                 max_price=20,  # 最大价格
                 interval_vol_ratio=[1.5, 5],  # 量比
                 interval_turnover_rate=[7, 15],  # 换手率
                 interval_pct_change=[2, 5],
                 threshs=[500000000, 10000000000],  # 市值
                 ):  #
    """
    目前只根据市值选股票
    Args:
        tushare_licence:
        threshs:

    Returns:

    """
    cur_date = datetime.now().strftime('%Y%m%d')
    save_path = os.path.join("tushare", cur_date)
    os.makedirs(save_path, exist_ok=True)
    ts.set_token(tushare_licence)
    df = ts.realtime_list(src='dc')

    final_stocks = []

    for i in range(len(df)):
        data = df.iloc[i].tolist()
        total_mv = data[16]
        code = data[0]
        dm, jys = code.split(".")
        jys = jys.lower()

        price = data[3]  # 当前价格
        if jys != jiaoyisuo:continue
        if "bj" in jys: continue  # 去掉北京的股票

        if "688" in dm or "300" in dm or "301" in dm: continue  # 跳过创业板等不能买的股票
        if price > max_price * 1.1: continue  # 过滤本金不足的股票
        if total_mv < threshs[0] and total_mv > threshs[1]: continue  # 选择市值在5亿到100亿的股票


        vol_ratio = data[12]
        turnover_rate = data[13]
        pct_change = data[3]
        if vol_ratio < interval_vol_ratio[0] or vol_ratio > interval_vol_ratio[1]: continue
        if turnover_rate < interval_turnover_rate[0] or turnover_rate > interval_turnover_rate[1]: continue
        if pct_change < interval_pct_change[0] or pct_change > interval_pct_change[1]: continue
        final_stocks.append(code)

        info = open(os.path.join(save_path, f"resul_{jiaoyisuo}.txt"), "a+")
        info.write(code + "\n")
        info.close()


    print(final_stocks)



if __name__ == '__main__':
    tushare_licence = "8f7419cb3d0bc956544a972de149e0e4263895e9467fef74f09b21c2"
    biying_licence = "42FC31E3-6DE7-43A8-B596-B865257A1025"
    save_root = "/home/algo/disk/project/code/coding/diffusion_coding/0814/gupiao/gupiao/gupiao/"
    log_path = os.path.join(save_root,"log.txt")
    logger = get_logger(log_path, mode="a" )

    while 1:
        current_min = datetime.now().strftime('%H:%M')
        if (datetime.strptime(current_min, "%H:%M") <= datetime.strptime("15:00", "%H:%M")
            and datetime.strptime(current_min, "%H:%M") >= datetime.strptime("13:00", "%H:%M")) \
                or (datetime.strptime(current_min, "%H:%M") <= datetime.strptime("11:30", "%H:%M")
                    and datetime.strptime(current_min, "%H:%M") >= datetime.strptime("9:00", "%H:%M")) :
            writing = True
        else:
            writing = False
        if writing:
            realtime_writing(tushare_licence,save_root)
            logger.info(f"writing {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        else:
            logger.info(f"sleep   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        time.sleep(60*5)
        # print(current_min)


    # choose_stock(tushare_licence,jiaoyisuo="sh")
    # choose_stock(tushare_licence,jiaoyisuo="sz")
