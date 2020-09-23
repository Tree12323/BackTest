from Backtest import  Backtest
import numpy as np
from FinantialStrategy import FinantialStrategy
from util import  interpolation
from PBO import PBO

from util import interpolation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

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
        self.pbo = PBO(all_strategy,10)
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
        for _ in range(self.N):
            data_new = []
            for i in range(self.data.shape[0]):
                idx = self.get_list(100, origin_length*2 - 1)
                while len(idx) < origin_length - 100:
                    idx = self.get_list(100, origin_length*2-1)
                idx = idx[:origin_length - 100]
                data_new.append(interpolation(self.data[i], 0)[idx, :])
            pbo, id_opt = self.pbo.calcPBO(np.array(data_new))
            score[id_opt] = score[id_opt] + 1 / (pbo + self.eps)
            print('PBO: ', pbo, 'id_opt: ', id_opt)
        id_opt = np.argmax(score)
        return id_opt


    # 输出一组生成的虚拟数据和原始数据
    def get_allData(self):
        origin_length = self.data.shape[1]
        data_new = np.zeros([self.N, self.data.shape[0], origin_length, 5])
        for n in range(self.N):
            item = np.zeros([self.data.shape[0], origin_length, 5])
            for i in range(self.data.shape[0]):
                idx = self.get_list(100, origin_length * 2 - 1)
                while len(idx) < origin_length:
                    idx = self.get_list(100, origin_length * 2 - 1)
                idx = idx[:origin_length]
                item[i] = (interpolation(self.data[i,:,:5], 3)[idx, :])
                item[i,:,1] = np.max(item[i,:,:3],1)
                item[i,:,3] = np.min(np.concatenate([item[i,:,0].reshape(-1,1),item[i,:,2:4]],1), 1)
            data_new[n, :, :, :] = item
        # data_new = np.zeros([self.N, self.data.shape[0], origin_length, 6])
        # for n in range(self.N):
        #     item = np.zeros([self.data.shape[0], origin_length, 6])
        #     for i in range(self.data.shape[0]):
        #         idx = self.get_list(100, origin_length * 2 - 1)
        #         while len(idx) < origin_length:
        #             idx = self.get_list(100, origin_length * 2 - 1)
        #         idx = idx[:origin_length]
        #         item[i] = (interpolation(np.concatenate([self.data[i,:,:5],self.data[i,:,-1].reshape(-1,1)],1) , 3)[idx, :])
        #         item[i,:,1] = np.max(item[i,:,:3],1)
        #         item[i,:,3] = np.min(np.concatenate([item[i,:,0].reshape(-1,1),item[i,:,2:4]],1), 1)
        #         print(i)
        #     data_new[n, :, :, :] = item
        data_new = data_new.reshape(-1,data_new.shape[-2], data_new.shape[-1])
        per = np.random.permutation(data_new.shape[0])  # 打乱后的行号
        # data_random0 = np.random.randn(self.data.shape[0],self.data.shape[1],self.data.shape[2])*np.std(self.data,1).reshape(len(self.data),1,-1)+np.mean(self.data,1).reshape(len(self.data),1,-1)
        # data_random1 = np.random.randn(self.data.shape[0],self.data.shape[1],self.data.shape[2])*np.std(self.data,1).reshape(len(self.data),1,-1)+np.mean(self.data,1).reshape(len(self.data),1,-1)
        # return data_random0[:,:,:5], self.data[:,:,:5]
        return data_new[per[:len(self.data)],:,:5], self.data[:,:,:5]

    # 检验虚拟数据能否被分类器分辨
    def validate(self):
        data_new, data_original = self.get_allData()
        data_new_f, data_original_f = np.fft.fft(data_new).real, np.fft.fft(data_original).real
        data = np.vstack((data_new, data_original))
        data_f = np.vstack((data_new_f, data_original_f))
        train = np.hstack((np.mean(data, 1), np.var(data, 1), np.max(data, 1), np.min(data, 1),
                           np.max(data, 1) - np.min(data, 1), np.max(data, 1) / np.mean(data, 1), np.mean(data_f, 1),
                           np.var(data_f, 1)))
        test = np.array([0] * len(data_new) + [1] * len(data_original))
        X_train, X_test, Y_train, Y_test = train_test_split(train, test, test_size=0.2, shuffle=True)
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        print('KNN score', model.score(X_test, Y_test))
        model = SGDClassifier()
        model.fit(X_train, Y_train)
        print('SVM score', model.score(X_test, Y_test))
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        print('DecisionTree score', model.score(X_test, Y_test))
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        print('RandomForest score', model.score(X_test, Y_test))