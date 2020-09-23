import numpy as np
from KPI import KPI

def calc(inp):
    return inp[:, 9]

def gap(open_price, close_price, init_money):
    return 1.0 * (close_price / open_price - 1) * init_money

def gap_colume(open_price, close_price, colume):
    return 1.0 * (close_price - open_price) * colume

def RSI(data, paras, standard_data_file):
    TIME_PERIOD = 14
    HIGH_RSI = 85
    LOW_RSI = 30
    ORDER_PERCENT = 0.3

    money = paras['InitMoney']
    cash = money

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

    NrOfShare = data.shape[0]
    hold_colume = np.zeros(NrOfShare, 'float32')
    length = np.zeros(NrOfShare, 'float32')
    p_pre = np.zeros(NrOfShare, 'float32')
    for i in range(data.shape[1]):
        if i < 14:
            continue
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

        RSI_val = data[:, i-13:i+1, 2] - data[:, i-14:i, 2]
        RSI_positive = []
        for j in range(RSI_val.shape[0]):
            RSI_positive.append(np.sum(RSI_val[j, RSI_val[j,:] > 0]))
        RSI_positive = np.array(RSI_positive)
        RSI_negative = []
        for j in range(RSI_val.shape[0]):
            RSI_negative.append(np.sum(RSI_val[j, RSI_val[j, :] < 0]))
        RSI_negative = np.array(RSI_negative)

        sell_share = RSI_positive / (RSI_positive - RSI_negative) * 100 > HIGH_RSI
        buy_share = RSI_positive / (RSI_positive - RSI_negative) * 100 < LOW_RSI

        hold_index = hold_colume > 0
        str_cur_close = data[hold_index, i - 1, 2]
        str_pre_close = data[hold_index, i, 2]
        str_gap_money = gap_colume(str_pre_close, str_cur_close, hold_colume[hold_index])
        str_money += np.sum(str_gap_money)
        strategy_daily_reward.append(np.sum(str_gap_money))
        if np.sum(hold_index) != 0:
            strategy_daily_ratio.append(1.0 * np.mean((str_cur_close - str_pre_close) / str_pre_close))
        else:
            strategy_daily_ratio.append(0)

        if np.sum(buy_share) > 0 and cash > 100:
            money_each_share = cash // np.sum(buy_share)
            hold_colume[buy_share] += money_each_share // (data[buy_share, i, 2] * 100) * 100
            cash -= np.sum(money_each_share // (data[buy_share, i, 2] * 100) * 100 * data[buy_share, i, 2])

        if np.sum(sell_share) > 0:
            sell_index = hold_index & sell_share
            cash += np.sum(hold_colume[sell_index] * data[sell_index, i, 2])
            hold_colume[sell_share] = np.zeros(np.sum(sell_share))

        p_pre = calc(data[:, i, :])
        std_cur_open = std_cur_close
        N = data.shape[1]

    for i in range(500 - N):
        npzero = np.array([0.0])
        strategy_daily_reward = np.append(npzero, strategy_daily_reward)
        strategy_daily_ratio = np.append(npzero, strategy_daily_ratio)
        standard_daily_reward = np.append(npzero, standard_daily_reward)
        standard_daily_ratio = np.append(npzero, standard_daily_ratio)

    N -= TIME_PERIOD
    return start_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio

if __name__ == '__main__':
    data = np.load('../saved files/data_zjz.npy')[:, :500, :]
    standard_data = '../saved files/standard_data.npy'
    init_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio = RSI(
        data, {'InitMoney': 1000000}, standard_data)
    for i in range(500 - len(strategy_daily_reward)):
        npzero = np.array([0.0])
        strategy_daily_reward = np.append(npzero, strategy_daily_reward)
        strategy_daily_ratio = np.append(npzero, strategy_daily_ratio)
        standard_daily_reward = np.append(npzero, standard_daily_reward)
        standard_daily_ratio = np.append(npzero, standard_daily_ratio)
    print('init_money shape:{}'.format(init_money))
    print('str_money shape:{}'.format(str_money))
    print('std_money shape:{}'.format(std_money))
    print('N shape:{}'.format(N))
    print('strategy_daily_reward shape:{}'.format(np.array(strategy_daily_reward).shape))
    print('strategy_daily_ratio shape:{}'.format(np.array(strategy_daily_ratio).shape))
    print('standard_daily_reward shape:{}'.format(np.array(standard_daily_reward).shape))
    print('standard_daily_ratio shape:{}'.format(np.array(standard_daily_ratio).shape))
    kpi = KPI(
        init_money=init_money,
        str_money=str_money,
        std_money=std_money,
        N=N,
        strategy_daily_reward=strategy_daily_reward,
        strategy_daily_ratio=strategy_daily_ratio,
        standard_daily_reward=standard_daily_reward,
        standard_daily_ratio=standard_daily_ratio
    )
    all_filed = kpi.get_kpi()
    money1 = 1000000.0
    money2 = 1000000.0
    daily_reward1 = strategy_daily_reward
    daily_reward2 = standard_daily_reward
    str_daily_reward_list = []
    std_daily_reward_list = []

    for i in range(len(daily_reward1)):
        money1 += daily_reward1[i]
        str_daily_reward_list.append(money1)

    for i in range(len(daily_reward2)):
        money2 += daily_reward2[i]
        std_daily_reward_list.append(money2)
    print(str_daily_reward_list)
    print(std_daily_reward_list)
    daily = []
    daily.append(np.array(str_daily_reward_list))
    daily.append(np.array(std_daily_reward_list))
    np.save('../saved files/strategy_0_daily.npy', np.array(daily))