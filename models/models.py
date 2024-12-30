#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2024/12/8 17:45
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os

import torch

from torch import nn

import torch.nn.functional as F

from models.diffusion import TransformerConcatLinear,DiffusionTraj,VarianceSchedule


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(Linear, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out

class PromptFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_vol = Linear(240,128,residual=True)
        self.linear_turn = Linear(5,128,residual=True)
        self.linear_pt = Linear(192, 128, residual=True)
        self.fusion = Linear(128*3,192)

    def forward(self,vol,turn,pt):
        vol = self.linear_vol(vol)
        turn = self.linear_turn(turn)
        pt = self.linear_pt(pt)
        return self.fusion(torch.cat([vol,turn,pt],1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffnet = TransformerConcatLinear(context_dim=5*48, tf_layer=3,
                                               residual=False,out_dim=1,
                                               nhead = 2)  # self.ego_pred_len*2)
        self.diffusion = DiffusionTraj(
            net=self.diffnet,
            var_sched=VarianceSchedule(num_steps=1000, beta_T=5e-2, mode='linear')
        )
        self.prompt = PromptFusion()
        self.out = Linear(192,192)

    def loss(self,gt,pt):
        loss = F.mse_loss(gt,pt,reduction='mean')
        return loss

    def generate(self, encoded_x, num_points, sample, bestof, point_dim,flexibility=0.0, ret_traj=False, sampling="ddpm",
                 step=1000):
        if sampling == "ddpm":
            predicted_y_vel_all,_ =  self.diffusion.sample(num_points, encoded_x, sample, bestof, point_dim=point_dim,flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        elif sampling == "ddim":
            predicted_y_vel_all,_ =  self.diffusion.sample_ddim(num_points, encoded_x, sample, bestof,point_dim=point_dim, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)

        return predicted_y_vel_all

    def forward(self,history_price,history_vol,history_turn,future_price,val=False):
        # print(history_price.max(),history_vol.max(),history_turn.max(),future_price.max())
        # print(history_price.min(), history_vol.min(), history_turn.min(), future_price.min())
        bs,seq,num = history_price.shape
        history_price = history_price.reshape(bs,1,-1)
        future_price = future_price.reshape(bs,-1,1)
        history_vol = history_vol.reshape(bs,-1)
        history_turn = history_turn.reshape(bs,-1)

        pt_hisprice = self.diffusion.get_loss(future_price,history_price)

        prompt = self.prompt(history_vol, history_turn,pt_hisprice.reshape(bs,-1))

        pt = self.out(pt_hisprice.reshape(bs,-1) + prompt)
        loss = self.loss(future_price.reshape(bs,-1),pt)
        if val:
            sample = 6
            with torch.no_grad():
                diff_sample = self.generate(history_price, num_points=192,point_dim=1,
                                            sample=sample, bestof=True,
                                            flexibility=0.0, ret_traj=False, sampling="ddim", step=100)
            pt_is = []
            for i in range(sample):
                pt_i = self.out(diff_sample[i].reshape(bs,-1) + prompt)
                pt_is.append(pt_i)
            pt_is_ = torch.stack(pt_is, 0)
        else:
            pt_is_ = None
        return  pt,pt_is_,loss



class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffnet = TransformerConcatLinear(context_dim=5*48, tf_layer=3,
                                               residual=False,out_dim=48,
                                               nhead = 2)  # self.ego_pred_len*2)
        self.diffusion = DiffusionTraj(
            net=self.diffnet,
            var_sched=VarianceSchedule(num_steps=1000, beta_T=5e-2, mode='linear')
        )
        self.prompt = PromptFusion()
        self.out = Linear(192,192)

    def loss(self,gt,pt):
        loss = F.mse_loss(gt,pt,reduction='mean')
        return loss

    def generate(self, encoded_x, num_points, sample, bestof, point_dim,flexibility=0.0, ret_traj=False, sampling="ddpm",
                 step=1000):
        if sampling == "ddpm":
            predicted_y_vel_all,_ =  self.diffusion.sample(num_points, encoded_x, sample, bestof, point_dim=point_dim,flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        elif sampling == "ddim":
            predicted_y_vel_all,_ =  self.diffusion.sample_ddim(num_points, encoded_x, sample, bestof,point_dim=point_dim, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)

        return predicted_y_vel_all


    def forward(self,history_price,history_vol,history_turn,future_price,val=False):
        # print(history_price.max(),history_vol.max(),history_turn.max(),future_price.max())
        # print(history_price.min(), history_vol.min(), history_turn.min(), future_price.min())
        bs,seq,num = history_price.shape
        # history_price = history_price #history_price.reshape(bs,1,-1)
        # future_price = future_price #future_price.reshape(bs,-1,1)
        history_vol = history_vol.reshape(bs,-1)
        history_turn = history_turn.reshape(bs,-1)

        pt_hisprice = self.diffusion.get_loss(future_price,history_price)

        prompt = self.prompt(history_vol, history_turn,pt_hisprice.reshape(bs,-1))

        pt = self.out(pt_hisprice.reshape(bs,-1) + prompt)
        loss = self.loss(future_price.reshape(bs,-1),pt)
        if val:
            sample = 6
            with torch.no_grad():
                diff_sample = self.generate(history_price, num_points=4,point_dim=48,
                                            sample=sample, bestof=True,
                                            flexibility=0.0, ret_traj=False, sampling="ddim", step=100)
            pt_is = []
            for i in range(sample):
                pt_i = self.out(diff_sample[i].reshape(bs,-1) + prompt)
                pt_is.append(pt_i)
            pt_is_ = torch.stack(pt_is, 0)
        else:
            pt_is_ = None
        return  pt,pt_is_,loss
if __name__ == '__main__':
    # 历史：价格 5 * 48，
    # 提示：成交量 5 * 48 ，历史换手率 5
    # 未来：价格 4 * 48
    model = Model()
    history_price = torch.ones((1,5,48),dtype=torch.float32)
    model(history_price)


