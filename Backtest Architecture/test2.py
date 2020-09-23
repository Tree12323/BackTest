import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
import pandas as pd
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

def plot_all(data_list, title, strategy):
    colors = ['red', 'blue', 'green', 'orange', 'black', 'pink', 'yellow']
    legend = [
        ['Real', 'interpolation_worst', 'interpolation_best', 'resample_worst', 'resample_best'],
        ['Real', 'interpolation_best', 'interpolation_worst', 'resample_best', 'resample_worst']
    ]
    plt.style.use('seaborn-darkgrid')
    for i in range(len(data_list)):
        data = np.array(data_list[i])
        x = range(data.shape[0])
        plt.plot(x, data, c=colors[i], label=legend[strategy][i])
    plt.legend()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.show()


def pick(daily_data, sum_data, all_data):
    # (10, 2, 500, 4)
    HMM_all_field_daily_data = daily_data
    # (10, 2, 11)
    HMM_all_field_sum_data = sum_data
    # (10, 300, 500, 13)
    HMM_all_stock = all_data

    strategy_num = HMM_all_field_sum_data.shape[1]
    strategy_id = []

    max_universe_total_reward_index = []
    min_universe_total_reward_index = []

    max_daily_data = []
    min_daily_data = []

    for i in range(strategy_num):
        strategy_id.append(i)
        # print(HMM_all_field_sum_data[:, i, 0])
        max_index = np.argmax(HMM_all_field_sum_data[:, i, 0])
        max_universe_total_reward_index.append(max_index)
        min_index = np.argmin(HMM_all_field_sum_data[:, i, 0])
        min_universe_total_reward_index.append(min_index)
        # test = HMM_all_field_daily_data[max_index][i].T
        max_daily_data.append(HMM_all_field_daily_data[max_index][i].T)
        min_daily_data.append(HMM_all_field_daily_data[min_index][i].T)


    # 返回值：策略（1， 2），每日数据
    return strategy_id, max_universe_total_reward_index, min_universe_total_reward_index, max_daily_data, min_daily_data

def get_return_reward(daily_list, init_money=1000000):
    tmp = init_money
    list = []
    for i in range(daily_list.shape[0]):
        tmp += daily_list[i]
        list.append(tmp)
    return np.array(list)

def recon_data(k_data):
    # require [date, open, close, high, low, volume]
    k_data[:, [1, 2]] = k_data[:, [2, 1]]
    tmp = range(500)
    data = np.c_[tmp, k_data]
    pd_data = pd.DataFrame(data)
    pd_data[0] = pd_data[0].astype('int')
    real_data = pd_data.to_numpy()
    return real_data


def plot_k(data, title):
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(bottom=0.5)
    mpf.candlestick_ochl(ax, data, width=0.3, colorup='g', colordown='r', alpha=1.0)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()



if __name__ == '__main__':
    # # 真实
    # RIS_file = '../saved files/strategy_0_reward.txt'
    # Self_file = '../saved files/Self.txt'
    real_k_file = '../saved files/data_zjz.npy'
    k_data = np.load(real_k_file, allow_pickle=True)
    k_data = k_data[1, :500, 0:5]
    k_data = recon_data(k_data)
    # ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20']
    plot_k(k_data, 'Real')



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

    # data0 = []
    # data0.append(np.loadtxt(RIS_file))
    # data0.append(np.loadtxt(Self_file))
    #
    # data1 = pick(
    #     daily_data=HMM_all_field_daily_data,
    #     sum_data=HMM_all_field_sum_data,
    #     all_data=HMM_all_stock,
    # )
    # data2 = pick(
    #     daily_data=interpolation_all_field_daily_data,
    #     sum_data=interpolation_all_field_sum_data,
    #     all_data=interpolation_all_stock,
    # )
    # data3 = pick(
    #     daily_data=resample_all_field_daily_data,
    #     sum_data=resample_all_field_sum_data,
    #     all_data=resample_all_stock,
    # )
    #
    # title0 = 'strategy-0'
    # data_list0 = []
    # data_list0.append(data0[0])
    # # data_list1.append(get_return_reward(data1[3][1][0]))
    # # data_list1.append(get_return_reward(data1[4][1][0]))
    # data_list0.append(get_return_reward(data2[3][0][0]))
    # data_list0.append(get_return_reward(data2[4][0][0]))
    # data_list0.append(get_return_reward(data3[3][0][0]))
    # data_list0.append(get_return_reward(data3[4][0][0]))
    #
    # plot_all(data_list0, title0, 0)
    #
    #
    # title1 = 'strategy-1'
    # data_list1 = []
    # data_list1.append(data0[1])
    # # data_list1.append(get_return_reward(data1[3][1][0]))
    # # data_list1.append(get_return_reward(data1[4][1][0]))
    # data_list1.append(get_return_reward(data2[3][1][0]))
    # data_list1.append(get_return_reward(data2[4][1][0]))
    # data_list1.append(get_return_reward(data3[3][1][0]))
    # data_list1.append(get_return_reward(data3[4][1][0]))
    #
    # plot_all(data_list1, title1, 1)

    for i in range(interpolation_all_stock.shape[0]):
        title = 'interpolation-%d' % (i)
        data1 = interpolation_all_stock[i, 1, :, 0:5]
        data1 = recon_data(data1)
        plot_k(data1, title)

    for i in range(resample_all_stock.shape[0]):
        title = 'resample-%d' % (i)
        data2 = interpolation_all_stock[i, 1, :, 0:5]
        data2 = recon_data(data2)
        plot_k(data2, title)




    # print(interpolation_all_stock.shape)



    # plot_Self(Join_file, Self_file)
