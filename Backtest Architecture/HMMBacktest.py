from Backtest import  Backtest
import numpy as np
from FinantialStrategy import FinantialStrategy
from util import  interpolation
from PBO import PBO
from hmmlearn import hmm
import numpy as np
import pandas as pd


class HMMBacktest(Backtest):
    """
        @:param:
        N:采样次数
        S:表示PBO的分割份数
    """

    def __init__(self, all_strategy, data,N, S,eps=1e-5):
        Backtest.__init__(self,all_strategy, data)
        self.N = N
        self.eps = eps
        self.S = S
        self.pbo = PBO(all_strategy,S)
        self.sample_data = np.zeros((N,data.shape[0],data.shape[1],data.shape[2]))
        self.sample_data_new = self.hmm_sample(data)

        self.all_field_daily_data = np.zeros([self.N, self.data.shape[1], 4])
        self.all_field_sum_data = np.zeros([self.N, 11])
        print(self.pbo.S)

        # 得到在当前分布下的的一次采样


    def hmm_sample(self,data, n_components=3, n_iter=2, sample_num=50, sample_len=100):
        stock_num,data_num = data.shape[0],data.shape[1]
        # 随机选取数据段

        res =  np.zeros([stock_num,self.N,data_num])

        for stock_item in range(stock_num):
            Sample = np.zeros(shape=(sample_num, sample_len))
            for i in range(sample_num):
                start = np.random.randint(0, data_num - sample_len)
                sample = data[stock_item,start:start + sample_len,2]
                # print(Sample.shape)
                # print(sample.shape)
                Sample[i] = sample
            Sample = Sample.reshape(-1, 1)
            # print("!!")
            # plt.plot(data)
            model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)  # startprob_prior = np.array([0.3,0.4,0.3]),
            model.fit(Sample, lengths=[sample_len] * sample_num)
            for item in range(self.N):
                res_item = model.sample(n_samples=data_num)[0]
                res[stock_item,item,:]= np.squeeze(res_item)

        res = res.transpose(1,0,2)
        self.sample_data[:,:,:,2] = res
        self.sample_data[:,:,:,0:2]=self.data[:,:,0:2]
        self.sample_data[:, :, :, 3:] = self.data[:, :, 3:]
        # print(self.sample_data.shape)
        # print("-----",self.sample_data[0,:,:,:].shape)
        return self.sample_data



    def do_Backtest(self):
        score = np.zeros(len(self.all_strategy), 'float32')
        origin_length = self.data.shape[1]

        for j in range(self.N):
            pbo, id_opt, self.all_field_daily_data[j, :, :], self.all_field_sum_data[j, :] = self.pbo.calcPBO(
                np.array(self.sample_data_new[j]))
            score[id_opt] = score[id_opt] + 1 / (pbo + self.eps)
            print('PBO: ', pbo, 'id_opt: ', id_opt)
        id_opt = np.argmax(score)
        return id_opt

    def get_backtest_data(self):
        return self.all_field_daily_data, self.all_field_sum_data