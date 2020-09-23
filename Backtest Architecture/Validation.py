from PBO import PBO
from Backtest import Backtest
import numpy as np
from InterpolationBackTest import InterpolationBackTest
from ReSampleBacktest import  ReSampleBacktest
from HMMBacktest import HMMBacktest


def validation(data,time,res,paras_list):
    if time ==0:
        return res
    # data = np.load('../saved files/data_zjz.npy')[:, :600, :]
    length = data.shape[1]
    train_data = data[:,:length-100, :]
    # print(train_data.shape)
    dev_data = data[:, length-100:, :]

    PBO_S = 2
    N = 1

    all_backtest = {}
    all_backtest["InterpolationBackTest"] = (InterpolationBackTest(paras_list, train_data, N, PBO_S, eps=1e-5))
    all_backtest["ReSampleBacktest_stock"] = ReSampleBacktest(paras_list, train_data, N, PBO_S)
    all_backtest["ReSampleBacktest_time"] = ReSampleBacktest(paras_list, train_data, N, PBO_S, type="stock")
    all_backtest["HMMBacktest"] = HMMBacktest(paras_list, train_data, N, PBO_S)


    basic_test  = Backtest(paras_list, data)
    _, id_opt, _, _, _ = basic_test.do_Backtest(train_data)
    print("using: basic   :",id_opt)

    all_choice = {}
    all_choice["basic"] = id_opt
    for key, back in all_backtest.items():
        print('using: ', key, " :")
        all_choice[key] = back.do_Backtest()
        print(back.get_backtest_data()[0].shape,back.get_backtest_data()[1].shape)



    _, id_opt, money_list, _, _ = basic_test.do_Backtest(dev_data)

    for key, choice in all_choice.items():
        print('using ',key,'you can get money in future days: ', money_list[choice])
        res[key] += money_list[choice]

    return validation(train_data,time-1, res,paras_list)


def main():
    InitMoney = 10000.0
    UpLength_list = [2]
    paras_list = []
    for UpLength in UpLength_list:
        paras_list.append({'InitMoney': InitMoney, 'UpLength': UpLength})


    data = np.load('../saved files/data_zjz.npy')[:, :600, :]
    res ={}
    res["InterpolationBackTest"]=0
    res["ReSampleBacktest_stock"]=0
    res["ReSampleBacktest_time"] = 0
    res["HMMBacktest"]=0
    res["basic"]=0
    print(validation(data,4,res,paras_list))

if __name__ == "__main__":
    main()

