#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2024/12/8 09:10
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
import numpy as np
import torch
from scipy.signal import hilbert2
from torch.utils.data import DataLoader

from utils1 import get_time_period_data


from tqdm import tqdm

# 分钟：收盘价，成交量
# 日线：换手率

class Data(torch.utils.data.Dataset):
    def __init__(self, data_root, scale_path = None,history_days=5, future_days=4,is_train=True,is_debug = False):
        super().__init__()

        self.history_days = history_days
        self.future_days = future_days
        day_data_list = []
        min_data_list = []
        stock_list = []

        if scale_path is None:
            self.scale = {}
        else:
            self.scale =np.load(scale_path, allow_pickle=True).item()

        folders = os.listdir(data_root)
        if is_train:
            folders = folders#[: int(len(folders) * 0.8)]
        else:
            folders = folders[int(len(folders) * 0.8):]

        if is_debug:
            folders = folders[:10]

        for folder in tqdm(folders):
            root = os.path.join(data_root, folder)
            if not os.path.isdir(root): continue
            names = sorted(os.listdir(root))
            if len(names) ==0:continue

            day_data, min_data = get_time_period_data([os.path.join(root, name) for name in names])
            price = min_data[:, :, 3].max()
            vol = min_data[:, :, 5].max()

            if scale_path is None:
                self.scale[folder] = [price, vol]

            for i in range(history_days, day_data.shape[0] - future_days):
                day_data_list.append(day_data[i - history_days:i + future_days])
                min_data_list.append(min_data[i - history_days:i + future_days])
                stock_list.append(folder)

        if scale_path is None:
            np.save("scale.npy",self.scale)


        self.day_data = day_data_list

        self.min_data = min_data_list

        self.stock = stock_list



    def __getitem__(self, index):
        stock = self.stock[index]
        day_datas, min_datas = self.day_data[index],self.min_data[index]
        prices = min_datas[:, :, 4]
        vols = min_datas[:, :, 5]
        turns = day_datas[:, 8]
        history_prices, future_prices = prices[:self.history_days], prices[self.history_days:]
        history_vols, future_vols = vols[:self.history_days], vols[self.history_days:]
        history_turns, future_turns = turns[:self.history_days], turns[self.history_days:]
        return torch.Tensor(history_prices * 2 / self.scale[stock][0] - 1), \
            torch.Tensor(future_prices* 2 / self.scale[stock][0] - 1), \
            torch.Tensor(history_vols* 2 / self.scale[stock][1] - 1), \
            torch.Tensor(future_vols* 2 / self.scale[stock][1] - 1), \
            torch.Tensor(history_turns * 2 / 100 - 1), \
            torch.Tensor(future_turns * 2 / 100 - 1)
    def __len__(self):
        return len(self.day_data)

if __name__ == '__main__':
    data = Data("/home/fb/project/gupiao_project/gupiao/gupiao_data_new")
    result = data.__getitem__(0)
    print(result)
