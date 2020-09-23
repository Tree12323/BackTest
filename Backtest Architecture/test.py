import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
def test1():
    csv_filename = '../saved files/result_1.csv'

    data = pd.read_csv(csv_filename, header=None).to_numpy()
    money = 1000000
    tmp1 = money
    tmp2 = money
    Join_data = data[:, 5]
    Self_data = data[:, 6]
    Join_return_data = []
    Self_return_data = []

    for i in range(488):
        tmp1 += Join_data[i]
        Join_return_data.append(tmp1)

    for i in range(500):
        tmp2 += Self_data[i]
        Self_return_data.append(str(tmp2))
    Join_return_data = np.array(Join_return_data).T
    Self_return_data = np.array(Self_return_data).T

    np.savetxt('../saved files/Join.txt', Join_return_data, fmt = '%s')
    np.savetxt('../saved files/Self.txt', Self_return_data, fmt = '%s')

def test3():
    a = np.array([[1, 2, 6], [2, 7, 3], [3, 9, 4]])
    index = np.unravel_index(a.argmax(), a.shape)
    print(index[1])

def test2():
    # (10, 2, 500, 4)
    HMM_all_field_daily_data = np.load('../saved files/total/HMM_all_field_daily_data.npz.npy')
    # (10, 2, 11)
    HMM_all_field_sum_data = np.load('../saved files/total/HMM_all_field_sum_data.npz.npy')
    # (10, 300, 500, 13)
    HMM_all_stock = np.load('../saved files/total/HMM_all_stock.npy')

    interpolation_all_field_daily_data = np.load('../saved files/total/interpolation_all_field_daily_data.npz.npy')
    # (10, 2, 11)
    interpolation_all_field_sum_data = np.load('../saved files/total/interpolation_all_field_sum_data.npz.npy')
    # (10, 300, 500, 13)
    interpolation_all_stock = np.load('../saved files/total/interpolation_all_stock.npy')

    resample_all_field_daily_data = np.load('../saved files/total/resample_all_field_daily_data.npz.npy')
    # (10, 2, 11)
    resample_all_field_sum_data = np.load('../saved files/total/resample_all_field_sum_data.npz.npy')
    # (10, 300, 500, 13)
    resample_all_stock = np.load('../saved files/total/resample_all_stock.npy')

    # max_01_index = np.unravel_index(HMM_all_field_sum_data[:, 0, 0].argmax(), HMM_all_field_sum_data.shape)
    strategy_num = HMM_all_field_sum_data.shape[1]
    strategy_id = []

    max_universe_total_reward_index = []
    min_universe_total_reward_index = []

    max_daily_data = []
    min_daily_data = []

    for i in range(strategy_num):
        strategy_id.append(i)
        # print(HMM_all_field_sum_data[:, i, 0])
        max_universe_total_reward_index.append(np.argmax(HMM_all_field_sum_data[:, i, 0]))
        min_universe_total_reward_index.append(np.argmin(HMM_all_field_sum_data[:, i, 0]))
        max_daily_data.append(HMM_all_field_daily_data[max_universe_total_reward_index][i])
        min_daily_data.append(HMM_all_field_daily_data[min_universe_total_reward_index][i])


    # 返回值：策略（1， 2），每日数据
    return strategy_id, max_universe_total_reward_index, min_universe_total_reward_index, max_daily_data, min_daily_data

def plot_Self(file1, file2):
    # data1 = pd.read_csv(file1, header=None).to_numpy()
    # data2 = pd.read_csv(file2, header=None).to_numpy()
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)
    label1 = range(data1.shape[0])
    label2 = range(data2.shape[0])
    plt.style.use('seaborn-darkgrid')
    # plt.plot(label1, data1, c='red')
    # plt.plot(label2, data2, c='blue')
    l1 = plt.plot(label1, data1)
    l2 = plt.plot(label2, data2)
    plt.legend((l1[0], l2[0]), ('JoinQuant', 'My_Data'))
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.show()
    print(plt.style.available)

def plot_k(data):

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(bottom=0.5)
    mpf.candlestick_ochl(ax, data, width=0.3, colorup='g', colordown='r', alpha=1.0)
    plt.grid(True)

    plt.title('K')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()



if __name__ == '__main__':
    # a = test2()
    # print(a[2])

    # Join_file = '../saved files/Join.txt'
    # Self_file = '../saved files/Self.txt'
    # plot_Self(Join_file, Self_file)

    file = '../saved files/data_zjz.npy'
    data = np.load(file, allow_pickle=True)
    # ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20']
    or_data = data[1, :500, 0:5]
    # require [date, open, close, high, low, volume]
    or_data[:, [1, 2]] = or_data[:, [2, 1]]
    tmp = range(500)
    data = np.c_[tmp, or_data]
    pd_data = pd.DataFrame(data)
    pd_data[0] = pd_data[0].astype('int')
    np_data = pd_data.to_numpy()
    print(pd_data.shape)
    # data[:, []]
    plot_k(np_data)

