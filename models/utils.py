# -*- coding: utf-8 -*-
# @File   : utils1.py
# @Time   : 2024/12/9 下午5:28 
# @Author : binge.van
# @decs   :


import os
import numpy as np
from matplotlib import pyplot as plt


def draw_min(pt, gt, his, w=800, h=300, message=""):
    pt = np.concatenate([his, pt], 0)
    gt = np.concatenate([his, gt], 0)
    seq_len, num = pt.shape
    pt = pt.reshape(-1)
    gt = gt.reshape(-1)
    day_boundaries, day_centers = \
        [num * i for i in range(seq_len)], \
        [num * i + num // 2 for i in range(seq_len)]
    x = np.linspace(1, len(pt), len(pt))
    plt.figure(figsize=(w / 100, h / 100))
    plt.title(message, fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    min_max = [min(min(pt), min(gt)),
               max(max(pt), max(gt))]  # [0,30] if "velocity" in message else [-3,7]
    plt.ylim(min_max)
    plt.plot(x, gt, label=f'gt')
    plt.plot(x, pt, label=f'pt')

    plt.legend(loc='upper left', fontsize=10, framealpha=0.5)
    for i in range(len(day_boundaries)):
        if i >= 6:
            c = "g"
            l = "--"
        elif i == 7:
            c = "b"
            l = "--"
        else:
            c = "r"
            l = "-"
        boundary = day_boundaries[i]
        plt.axvline(x=boundary, color=c, linestyle=l, linewidth=1)

    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=True)

    canvas = plt.gcf().canvas
    canvas.draw()
    plt_fig2 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)  # [200,200,3]
    img = plt_fig2[..., ::-1]
    return img


if __name__ == '__main__':
    pass
