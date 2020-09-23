import talib as ta
import numpy as np
import pandas as pd
from KPI import KPI

class strategy_factory:

    def create_strategy(strategy_name):
        strategy_class = globals()[strategy_name]
        return strategy_class

class golden_cross_strategy():

    def __init__(self, parameter_list, data):
        self.data = data
        self.strategy_name = "golden_cross_strategy"
        self.data_list = []
        self.std_bar_data = self.load_data(data)
        self.money = [float(parameter_list[0]) for i in range(len(data))]
        self.std_money = float(parameter_list[0])
        self.cur_position = [0.0 for i in range(len(data))]
        self.add_strategy_parameters(shortperiod=parameter_list[1], longperiod=parameter_list[2])
        self.max_run_days = len(data[0])

    def load_data(self, data):
        std_bar_data = pd.read_csv('../saved files/data.csv')
        self.add_data_list(data)
        return std_bar_data

    def add_data_list(self, data):
        for stock in data:
            self.data_list.append(stock)

    def add_strategy_parameters(self, shortperiod=5, longperiod=20):
        self.shortperiod = shortperiod
        self.longperiod = longperiod

    def run_all_stocks(self):
        max_gain_stockid = 0
        max_gain_money = 0.0
        money_list = []
        kpi = None

        for stock_id in range(len(self.data_list)):
            kpi = self.run_one_stock_maxdays(self.data_list[stock_id], stock_id)
            money_list.append(kpi.str_money)
            if max_gain_money < kpi.str_money:
                max_gain_money = kpi.str_money
                max_gain_stockid = stock_id

        return max_gain_stockid, max_gain_money, money_list, kpi

    def run_one_stock_maxdays(self, data, stock_id):
        total_money = 0.0
        std_money = 0.0
        strategy_reward_list = []
        strategy_ratio_list = []
        std_reward_list = []
        std_ratio_list = []

        for day_nums in range(self.max_run_days):
            total_money, std_money, day_time, strategy_reward, strategy_ratio, std_reward, std_ratio = self.run_one_day(day_nums, stock_id, data)
            strategy_reward_list.append(strategy_reward)
            strategy_ratio_list.append(strategy_ratio)
            std_reward_list.append(std_reward)
            std_ratio_list.append(std_ratio)

        kpi = KPI(
            init_money=10000.0,
            str_money=total_money,
            std_money=std_money,
            N=self.max_run_days - self.longperiod - 5,
            strategy_daily_reward=strategy_reward_list,
            strategy_daily_ratio=strategy_ratio_list,
            standard_daily_reward=std_reward_list,
            standard_daily_ratio=std_ratio_list
        )
        return kpi

    def run_one_day(self,day_nums, stock_id, data):
        if day_nums < 25:
            # shape 475 ~ shape 500
            # 25天补上【】【
            # return self.get_KPI_parameters(0, self.money[stock_id], 0, stock_id)
            return self.money, self.std_money, 0, 0, 0, 0, 0

        day_offset = day_nums - self.longperiod - 5
        prices = data[ day_offset: day_nums][:,2]

        short_avg = ta.SMA(prices, self.shortperiod)
        long_avg = ta.SMA(prices, self.longperiod)

        shares = self.money[stock_id] / prices[-1]

        original_money = self.money[stock_id] + self.cur_position[stock_id] * prices[-2]

        if short_avg[-1] - long_avg[-1] < 0 and short_avg[-2] - long_avg[-2] > 0 and self.cur_position[stock_id] > 0 :
            self.money[stock_id] += self.cur_position[stock_id] * prices[-1]
            self.cur_position[stock_id] = 0

        if short_avg[-1] - long_avg[-1] > 0 and short_avg[-2] - long_avg[-2] < 0:
            self.money[stock_id] -= shares * prices[-1]
            self.cur_position[stock_id] += shares

        return self.get_KPI_parameters(day_nums, original_money, prices[-1], stock_id)

    def get_KPI_parameters(self, days, original_money, close_money, stock_id):
        total_money = self.money[stock_id] + self.cur_position[stock_id] * close_money
        strategy_reward = total_money - original_money
        strategy_ratio = strategy_reward / original_money

        yesterday_close_std_price = self.std_bar_data[days-2:days-1]['close'].iloc[0]
        close_std_price = self.std_bar_data[days-1:days]['close'].iloc[0]
        std_ratio = (close_std_price - yesterday_close_std_price) / yesterday_close_std_price
        std_reward = std_ratio * self.std_money
        self.std_money *= std_ratio + 1

        return total_money, self.std_money, days, strategy_reward, strategy_ratio, std_reward, std_ratio


data = np.load('../saved files/data_zjz.npy')[97:98, :600, :]


""""
InitMoney = 10000.0
para_list = []
parameter_list = [InitMoney, 5, 20]
para_list.append(parameter_list)
data = np.load('/Users/wuchanglai/PycharmProjects/smartbacktest-1.0/saved\ files/data_zjz.npy ')[:, :600, :]
strategy_class = strategy_factory.create_strategy("golden_cross_strategy")
strategy = strategy_class(para_list[0], data)

print(strategy.run_all_stocks())
"""
