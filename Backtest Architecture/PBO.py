import numpy as np
from Backtest import Backtest

class PBO:
    def __init__(self,paras_list,S):
        self.paras_list = paras_list
        self.S = S
        self.data = None

    def next_list(self,cur_list):

        S2 = int(self.S / 2)
        for i in range(S2 - 1, -1, -1):
            if cur_list[i] < self.S - (S2 - i):
                cur_list[i] = cur_list[i] + 1
                for j in range(i + 1, S2):
                    cur_list[j] = cur_list[j - 1] + 1
                return cur_list
        return None

    def make_data(self,L):
        tmp = int(self.data.shape[1] / self.S)
        ret = []
        for l in L:
            ret.append(self.data[:, tmp * l:tmp * (l + 1), :])
        return np.concatenate(ret, 1)



    """
        @:param
            paras_list:all strategy
        @:return
            PBO value  
            choice(strategy) 
    """
    def calcPBO(self,data):
        self.data = data
        S2 = int(self.S/2)
        L = list(range(S2))
        numerator = 0
        denominator = 0
        back_test = Backtest(self.paras_list,data)

        while L is not None:
            L2 = list(set(list(range(self.S))) - set(L))
            _, id_opt, _, _, _ = back_test.do_Backtest(self.make_data(L))
            _, _, money_list, _, _ = back_test.do_Backtest(self.make_data(L2))
            if money_list[id_opt] < np.sort(money_list)[int(len(self.paras_list) / 2)]:
                numerator += 1
            denominator += 1
            L = self.next_list(L)
        _, id_opt, _, item_filed_daily_data, item_filed_sum_data = back_test.do_Backtest(data)

        return 1.0 * numerator / denominator, id_opt, item_filed_daily_data, item_filed_sum_data