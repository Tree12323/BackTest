import numpy as np
import random
from KPI import KPI

# 每日收益和每日收益率
def gap(open_price, close_price, init_money, num_stock):
    return 1.0 * (np.sum(close_price / open_price) - num_stock) * init_money / num_stock
def std_gap(open_price, close_price, init_money):
    return 1.0 * (close_price / open_price - 1) * init_money
def ratio(pre_money, cur_money):
    return 1.0 * (cur_money - pre_money) / pre_money

# 收益率
def calRR(pre_close, prepre_close):
    """
    :param pre_close: 昨天的收盘价
    :param prepre_close: (returnPeriod+1)天前的收盘价
    :return: 收益率
    """
    return 1.0 * (pre_close - prepre_close) / prepre_close

# 换仓检测
def getChangeNO(position_index, position_value, stock_pool_index, stock_pool_value, change_No, num_stock):
    position_part_index = position_index[-change_No:]
    position_part_value = position_value[-change_No:]
    stock_pool_part_index = stock_pool_index[0:change_No]
    stock_pool_part_value = stock_pool_value[0:change_No]

    part_index = np.append(position_part_index, stock_pool_part_index)
    part_value = np.append(position_part_value, stock_pool_part_value)

    tmp_index = np.argsort(-part_value)
    part1_index = part_index[tmp_index][0:change_No]
    part1_value = part_value[tmp_index][0:change_No]
    part2_index = part_index[tmp_index][change_No:]
    part2_value = part_value[tmp_index][change_No:]

    test1 = position_index[0:(num_stock-change_No)]
    test2 = part1_index
    test3 = position_value[0:(num_stock-change_No)]
    test4 = part1_value
    position_index = np.append(position_index[0:(num_stock-change_No)], part1_index)
    position_value = np.append(position_value[0:(num_stock-change_No)], part1_value)
    stock_pool_index = np.append(stock_pool_index[change_No:], part2_index)
    stock_pool_value = np.append(stock_pool_value[change_No:], part2_value)

    position_index, position_value, stock_pool_index, stock_pool_value = sortStock(
        position_index=position_index,
        position_value=position_value,
        stock_pool_index=stock_pool_index,
        stock_pool_value=stock_pool_value,
    )

    return position_index, position_value, stock_pool_index, stock_pool_value

# 收益率排序
def sortStock(position_index, position_value, stock_pool_index, stock_pool_value):
    position_tmp_index = np.argsort(-position_value)
    position_index = position_index[position_tmp_index]
    position_value = position_value[position_tmp_index]
    stock_pool_tmp_index = np.argsort(-stock_pool_value)
    stock_pool_index = stock_pool_index[stock_pool_tmp_index]
    stock_pool_value = stock_pool_value[stock_pool_tmp_index]
    return position_index, position_value, stock_pool_index, stock_pool_value

def AlpacaStrategy(init_money, data, chango_No=2, holdingPeriod=60, num_stock=20, std_data_file=None):

    std_data = np.load(std_data_file)

    # 股票池
    stock_pool_index = []
    stock_pool_value = []
    # 持仓
    position_index = []
    position_value = []
    # 初始资金
    start_money = init_money
    # 累计资金
    str_money = init_money
    std_money = init_money
    # 每日策略收益和基准收益
    standard_daily_reward = []
    strategy_daily_reward = []
    standard_daily_ratio = []
    strategy_daily_ratio = []

    # Step.1:随机取20支股票
    position_index = random.sample(range(300), num_stock)
    position_index = np.array(position_index)
    stock_pool_index = list(set(range(300)).difference(set(position_index)))
    stock_pool_index = np.array(stock_pool_index)


    for i in range(data.shape[1]):
        if i == 0:
            continue
        # 计算策略每日收益
        position_pre_close = data[position_index][:, i - 1, 2]
        position_cur_close = data[position_index][:, i, 2]
        str_gap_money = gap(
            open_price=position_pre_close,
            close_price=position_cur_close,
            init_money=str_money,
            num_stock=num_stock,
        )
        pre_str_money = str_money
        str_money += str_gap_money
        strategy_daily_reward.append(str_gap_money)
        strategy_daily_ratio.append(ratio(pre_str_money, str_money))

        # 计算基准每日收益
        std_cur_close = std_data[i][3]
        std_pre_close = std_data[i - 1][3]
        std_gap_money = std_gap(std_pre_close, std_cur_close, std_money)
        pre_std_money = std_money
        std_money += std_gap_money
        standard_daily_reward.append(std_gap_money)
        standard_daily_ratio.append(ratio(pre_std_money, std_money))

        # Step.2:holdingPeriod周期后计算收益率
        if i % holdingPeriod == 0:
            # 持仓收益率
            position_prepre_close = data[position_index][:, i - holdingPeriod, 2]
            position_value = calRR(position_pre_close, position_prepre_close)
            # 股票池收益率
            stock_pool_pre_close = data[stock_pool_index][:, i - 1, 2]
            stock_pool_prepre_close = data[stock_pool_index][:, i - holdingPeriod, 2]
            stock_pool_value = calRR(stock_pool_pre_close, stock_pool_prepre_close)

            # 排序
            position_index, position_value, stock_pool_index, stock_pool_value = sortStock(
                position_index=position_index,
                position_value=position_value,
                stock_pool_index=stock_pool_index,
                stock_pool_value=stock_pool_value,
            )

            # Step.3:换仓
            position_index, position_value, stock_pool_index, stock_pool_value = getChangeNO(
                position_index=position_index,
                position_value=position_value,
                stock_pool_index=stock_pool_index,
                stock_pool_value=stock_pool_value,
                change_No=chango_No,
                num_stock=num_stock,
            )
    N = data.shape[1]
    npzero = np.array([0.0])
    strategy_daily_reward = np.append(npzero, strategy_daily_reward)
    strategy_daily_ratio = np.append(npzero, strategy_daily_ratio)
    standard_daily_reward = np.append(npzero, standard_daily_reward)
    standard_daily_ratio = np.append(npzero, standard_daily_ratio)
    return start_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio
if __name__ == '__main__':
    init_money = 10000.0
    str_data = np.load('../saved files/data_zjz.npy')[:, :500, :]
    std_data_file = '../saved files/standard_data.npy'


    init_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio = AlpacaStrategy(
        init_money=init_money,
        data=str_data,
        std_data_file=std_data_file
    )


    # npzero = np.array([0.0])
    # strategy_daily_reward = np.append(npzero, strategy_daily_reward)
    # strategy_daily_ratio = np.append(npzero, strategy_daily_ratio)
    # standard_daily_reward = np.append(npzero, standard_daily_reward)
    # standard_daily_ratio = np.append(npzero, standard_daily_ratio)
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
    # 累积收益
    money = 1000000.0
    daily_reward = kpi.get_str_daily_rewards()
    daily_reward_list = []
    for i in range(len(daily_reward)):
        money += daily_reward[i]
        daily_reward_list.append(money)
    daily_reward_list.append(np.array(daily_reward_list))
    daily_reward_list.append(np.array(standard_daily_reward))

    np.savetxt('../saved files/strategy_1_reward.npy', np.array(daily_reward_list), fmt='%.2f')
    # test()
