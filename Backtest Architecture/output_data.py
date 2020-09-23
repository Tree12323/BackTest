from KPI import KPI
from FinantialStrategy import *
import numpy as np
from Backtest import Backtest


def output_npy():
    InitMoney = 10000.0
    UpLength_list = [2, 3, 4, 5, 6, 7]
    paras_list = []
    for UpLength in UpLength_list:
        paras_list.append({'InitMoney': InitMoney, 'UpLength': UpLength})
    data = np.load('../saved files/data_zjz.npy')[:, :600, :]
    length = data.shape[1]
    train_data = data[:, :length-100, :]
    standard_data_file = '../saved files/standard_data.npy'

    backtest = Backtest(
        paras_list=paras_list,
        data=train_data
    )
    backtest.do_Backtest(train_data)

def read_npy():
    data = np.load('../saved files/result/strategy-1.npy', allow_pickle=True)
    print(type(data[1]))

if __name__ == '__main__':
    read_npy()

