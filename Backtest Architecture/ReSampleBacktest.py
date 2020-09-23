from Backtest import  Backtest
import numpy as np
from FinantialStrategy import FinantialStrategy
from util import  interpolation
from PBO import PBO

class ReSampleBacktest(Backtest):
    """
        @:param:
        N:the new data
    """
    def __init__(self, all_strategy, data,N, S,type="stock",eps=1e-5):
        # self.all_strategy = all_strategy
        # self.data = data
        Backtest.__init__(self,all_strategy, data)
        self.N = N
        self.eps = eps
        self.S = S
        self.pbo = PBO(all_strategy,S)
        print(self.pbo.S)
        self.type = type
        # 时序分组
        self.time_num = 10

        self.all_field_daily_data = np.zeros([self.N, self.data.shape[1], 4])
        self.all_field_sum_data = np.zeros([self.N, 11])

        # 得到在当前分布下的的一次采样

    def do_Backtest(self):
        score = np.zeros(len(self.all_strategy), 'float32')
        for j in range(self.N):
            data = None
            if self.type=="stock":
                idx = np.random.choice(self.data.shape[0], self.data.shape[0])
                print(self.data[idx].shape)
                data= self.data[idx]
            elif self.type=="time":
                idx = np.random.choice(self.time_num, 10)
                datas = []
                time_lenth = self.data.shape[1]
                for id in idx:
                    datas.append(self.data[:,id*time_lenth/self.time_num:(id+1)*time_lenth/self.time_num,:])
                data = np.concatenate(datas,axis=1)

            pbo, id_opt, self.all_field_daily_data[j, :, :], self.all_field_sum_data[j, :] = self.pbo.calcPBO(
                np.array(data))
            score[id_opt] = score[id_opt] + 1 / (pbo + self.eps)
            print('PBO: ', pbo, 'id_opt: ', id_opt)
        id_opt = np.argmax(score)
        return id_opt

    def get_backtest_data(self):
        return self.all_field_daily_data, self.all_field_sum_data
