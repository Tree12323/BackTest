#the time in data is inverse and not continuous
import numpy as np
from FinantialStrategy import FinantialStrategy
from KPI import KPI
from AlpacaStrategy import AlpacaStrategy
class Backtest:
    """
        @:param:
        paras_list:all strategey
        data: all data

    """

    def __init__(self,paras_list, data):
        self.all_strategy = paras_list
        self.data = data
        self.standard_data_file = '../saved files/standard_data.npy'
        self.daily_filed = np.zeros([self.data.shape[1], 4])
        self.sum_filed = np.zeros([11])
    """
        @:param:
            data:
        @:return
            tuble(money_opt, id_opt, money_list)
            money_opt: the largetst  income
            id_opt:the choice(strategy)
            money_list:income list
    """
    def do_Backtest(self, data):
        # 输出文本
        money_opt = 0
        id_opt = None
        money_list = []
        # for i in range(len(self.all_strategy)):
        #     # money = FinantialStrategy(data, self.all_strategy[i])
        #     init_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio = FinantialStrategy(
        #         data, self.all_strategy[i], self.standard_data_file)
        #     kpi = KPI(
        #         init_money=init_money,
        #         str_money=str_money,
        #         std_money=std_money,
        #         N=N,
        #         strategy_daily_reward=strategy_daily_reward,
        #         strategy_daily_ratio=strategy_daily_ratio,
        #         standard_daily_reward=standard_daily_reward,
        #         standard_daily_ratio=standard_daily_ratio
        #     )
        #     all_filed = kpi.get_kpi()
        #     self.daily_filed = np.array(all_filed[0:4]).T
        #     self.sum_filed = np.array(all_filed[4:])
        #
        #
        #     if str_money>money_opt:
        #         money_opt = str_money
        #         id_opt = i
        #     money_list.append(str_money)
        init_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio = AlpacaStrategy(
            init_money=10000.0,
            data=data,
            std_data_file=self.standard_data_file,
        )

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
        self.daily_filed = np.array(all_filed[0:4]).T
        self.sum_filed = np.array(all_filed[4:])
        money_opt = str_money
        id_opt = 0
        money_list.append(str_money)
        return money_opt, id_opt, money_list, self.daily_filed, self.sum_filed

    def get_data(self):
        return self.daily_filed, self.sum_filed