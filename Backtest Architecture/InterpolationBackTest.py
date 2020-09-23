from Backtest import  Backtest
import numpy as np
from FinantialStrategy import FinantialStrategy
from util import  interpolation
from PBO import PBO

class InterpolationBackTest(Backtest):
    """
        @:param:
        N:采样次数
        S:表示PBO的分割份数
    """

    def __init__(self, all_strategy, data,N, S,eps=1e-5):
        # self.all_strategy = all_strategy
        # self.data = data
        Backtest.__init__(self,all_strategy, data)
        self.N = N
        self.eps = eps
        self.S = S
        self.pbo = PBO(all_strategy,S)

        self.all_field_daily_data = np.zeros([self.N, self.data.shape[1], 4])
        self.all_field_sum_data = np.zeros([self.N, 11])
        print(self.pbo.S)

        # 得到在当前分布下的的一次采样

    def get_list(self, steps, Num):
        p1 = np.array([1.0 / i for i in range(1, steps - 1)]).sum()
        p2 = 2 * p1
        # 得到所有的 概率因子
        p = np.array([p1, p2] + [1.0 / (i - 2) / (i - 2) for i in range(3, steps + 1)])
        # 归一化 得到概率
        p = p / p.sum()
        pos = 0
        result = [0]
        while pos < Num:
            val = np.random.rand()
            for i in range(steps):
                if p[:i + 1].sum() > val:
                    pos += i + 1
                    if pos < Num:
                        result.append(pos)
                    break
        return result

    def do_Backtest(self):
        score = np.zeros(len(self.all_strategy), 'float32')
        origin_length = self.data.shape[1]
        for j in range(self.N):
            data_new = []
            for i in range(self.data.shape[0]):
                idx = self.get_list(100, origin_length*2 - 1)
                while len(idx) < origin_length:
                    idx = self.get_list(100, origin_length*2-1)
                idx = idx[:origin_length]
                data_new.append(interpolation(self.data[i], 0)[idx, :])
            print(np.array(data_new).shape)
            pbo, id_opt, self.all_field_daily_data[j, :, :], self.all_field_sum_data[j, :] = self.pbo.calcPBO(
                np.array(data_new))
            score[id_opt] = score[id_opt] + 1 / (pbo + self.eps)
            print('PBO: ', pbo, 'id_opt: ', id_opt)
        id_opt = np.argmax(score)
        return id_opt


    def get_backtest_data(self):
        return self.all_field_daily_data, self.all_field_sum_data

