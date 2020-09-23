# import numpy as np
#
# def calc(inp):
#     return inp[:, 9]
#
# def FinantialStrategy(data, paras):
#     money = paras['InitMoney']
#     UpLength = paras['UpLength']
#     NrOfShare = data.shape[0]
#     length = np.zeros(NrOfShare, 'float32')
#     p_pre = np.zeros(NrOfShare, 'float32')
#     for i in range(data.shape[1]):
#         p_cur = calc(data[:, i, :])
#         tmp = 1.0*(p_cur>=p_pre)
#         length = length*tmp+tmp
#         if np.max(length)==UpLength:
#             length = np.zeros(NrOfShare, 'float32')
#             id_cur = np.argmax(length)
#             price = data[id_cur, i, 2]
#             nr_cur = money/price
#             for j in range(i+1, data.shape[1]):
#                 if j==data.shape[1]-1 or data[id_cur, j, 0]>=price+1:
#                     money = nr_cur*data[id_cur, j, 0]
#                     i = j
#                     break
#         p_pre = calc(data[:, i, :])
#     return money


import numpy as np

def calc(inp):
    return inp[:, 9]

def gap(open_price, close_price, init_money):
    return 1.0 * (close_price / open_price - 1) * init_money

def FinantialStrategy(data, paras, standard_data_file):
    money = paras['InitMoney']

    start_money = money
    # 累计资金
    str_money = money
    std_money = money
    # 基准数据
    standard_data = np.load(standard_data_file)
    # 每日策略收益和基准收益
    strategy_daily = []
    standard_daily_reward = []
    strategy_daily_reward = []
    standard_daily_ratio = []
    strategy_daily_ratio = []
    std_cur_open = standard_data[0][1]

    UpLength = paras['UpLength']
    NrOfShare = data.shape[0]
    length = np.zeros(NrOfShare, 'float32')
    p_pre = np.zeros(NrOfShare, 'float32')
    for i in range(data.shape[1]):

        # 基准收益计算
        std_cur_close = standard_data[i][3]
        # 计算基准每日收益
        std_gap_money = gap(std_cur_open, std_cur_close, init_money=std_money)
        # total——monry
        std_money += std_gap_money
        # 日收益放入到list
        standard_daily_reward.append(std_gap_money)
        # 收益率放到list
        standard_daily_ratio.append(1.0 * (std_cur_close - std_cur_open) / std_cur_open)

        p_cur = calc(data[:, i, :])
        tmp = 1.0*(p_cur>=p_pre)
        length = length*tmp+tmp
        if np.max(length)==UpLength:
            length = np.zeros(NrOfShare, 'float32')
            id_cur = np.argmax(length)

            strategy_daily.append(id_cur)
            str_cur_close = data[id_cur, i - 1, 2]
            str_pre_close = data[id_cur, i, 2]
            str_gap_money = gap(str_pre_close, str_cur_close, str_money)
            str_money += str_gap_money
            strategy_daily_reward.append(str_gap_money)
            strategy_daily_ratio.append(1.0 * (str_cur_close - str_pre_close) / str_pre_close)

            price = data[id_cur, i, 2]
            nr_cur = money/price
            for j in range(i+1, data.shape[1]):
                if j==data.shape[1]-1 or data[id_cur, j, 0]>=price+1:
                    money = nr_cur*data[id_cur, j, 0]
                    i = j
                    break
        else:
            strategy_daily_reward.append(0.0)
            strategy_daily_ratio.append(0.0)

        p_pre = calc(data[:, i, :])
        std_cur_open = std_cur_close
        N = data.shape[1]

    # print(start_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio)

    return start_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio
