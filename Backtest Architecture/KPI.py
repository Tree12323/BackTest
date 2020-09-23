import pandas as pd
import numpy as np

# 策略收益率
def total_returns(P_end, P_start):
    """
    :param P_end: 策略最终股票和现金的总价值
    :param P_start: 策略开始股票和现金的总价值
    :return: 策略收益
    """
    rtn = 1.0 * (P_end - P_start) / P_start
    return rtn

# 策略年化收益
def total_annualized_returns(P_rtn, N):
    """
    :param P_rtn: 策略收益
    :param N: 策略执行天数
    :return: 策略年化收益
    """
    R_p = pow((1 + P_rtn), 250 / N) - 1
    return R_p

# 贝塔 Beta
def beta(D_p, D_m, Cov_Dp_Dm=0, Var_Dm=0):
    """
    :param D_p: 策略每日收益
    :param D_m: 基准每日收益
    :param Cov_Dp_Dm: 策略每日收益与基准每日收益的协方差
    :param Var_Dm: 基准每日收益的方差
    :return: beta
    """
    D_p = np.array(D_p)
    D_m = np.array(D_m)
    Var_Dm = np.var(D_m)
    X = np.hstack((D_p, D_m))
    Cov_Dp_Dm = np.cov(X)
    B_p = 1.0 * Cov_Dp_Dm / Var_Dm
    return B_p

# 阿尔法 Alpha
def alpha(R_p, R_m, B_p, R_f=0.04):
    """
    :param R_p: 策略年化收益率
    :param R_f: 无风险利率（默认0.04）
    :param R_m: 基准年化收益率
    :param B_p: 策略beta值
    :return: Alpha值
    """
    a = R_p - (R_f + B_p * (R_m - R_f))
    return a

# 策略波动率 Algorithm Volatility
def sigma_p(r_p, r__p, N):
    """
    :param r_p: 策略每日收益率
    :param r__p: 策略每日收益率的平均值
    :param N: 策略执行天数
    :return: sigma_p
    """
    r__p_list = np.ones(len(r_p), dtype=float)
    r__p_list = r__p_list * r__p
    tmp = 1.0 * (250 / (N - 1)) * (np.linalg.norm(r_p - r__p_list) ** 2)
    sigma_p = np.sqrt(1.0 * (250 / (N - 1)) * (np.linalg.norm(r_p - r__p_list)) ** 2)
    return sigma_p

# 夏普比率 Sharpe Ratio
def sharpe_ratio(R_p, sigma_p, R_f=0.04):
    """
    :param R_p: 策略年化收益率
    :param R_f: 无风险利率（默认0.04）
    :param sigma_p: 策略收益波动率
    :return: sharpe_ratio
    """
    sharpe_ratio = 1.0 * (R_p - R_f) / sigma_p
    return sharpe_ratio

# 下行波动率 Downside Risk
def downside_risk(r_p, N):
    """
    :param r_p: 策略每日收益率
    :param r__pi: 策略至第i日平均收益率
    :param N: 策略执行天数
    :return: downside_risk
    """
    r__pi = []
    for i in range(len(r_p)):
        if i == 0:
            r__pi.append(r_p[0])
            continue
        r__pi.append(np.mean(r_p[0:i]))
    tmp = 0
    for i in range(N):
        if r_p[i] >= r__pi[i]:
            f_t = 0
        else:
            f_t = (r_p[i] - r__pi[i]) ** 2
        tmp += f_t

    downside_risk = np.sqrt(1.0 * (250 / N) * tmp)
    return downside_risk

# 最大回撤 Max Drawdown
def max_drawdown(init_money, strategy_daily_reward):
    """
    :param P_x: 策略x日股票和现金的总价值，y>x
    :param P_y: 策略y日股票和现金的总价值，y>x
    :return:
    """
    # max_drawdown = np.argmax(1.0 * (P_x - P_y) / P_x)
    str_return = []
    rtn = init_money

    for i in range(len(strategy_daily_reward)):
        rtn += strategy_daily_reward[i]
        str_return.append(rtn)

    gap_money = 0
    gap_start = 0

    for i in range(len(str_return) - 1):
        for j in range(i, len(str_return)):
            tmp = str_return[i] - str_return[j]
            if tmp >= gap_money:
                gap_money = tmp
                gap_start = str_return[i]

    Max_drawdown = 1.0 * gap_money / gap_start
    return Max_drawdown

class KPI:
    def __init__(self, init_money, str_money, std_money, N, strategy_daily_reward, strategy_daily_ratio, standard_daily_reward, standard_daily_ratio):
        self.str_money = str_money
        self.std_money = std_money
        self.N = N
        self.strategy_daily_reward = strategy_daily_reward
        self.strategy_daily_ratio = strategy_daily_ratio
        self.standard_daily_reward = standard_daily_reward
        self.standard_daily_ratio = standard_daily_ratio
        self.start_money = init_money

        self.str_total_returns = total_returns(
            P_end=self.str_money,
            P_start=self.start_money
        )
        self.str_total_annualized_returns = total_annualized_returns(
            P_rtn=self.str_total_returns,
            N=self.N
        )
        self.std_total_returns = total_returns(
            P_end=self.std_money,
            P_start=self.start_money
        )
        self.std_total_annualized_returns = total_annualized_returns(
            P_rtn=self.std_total_returns,
            N=self.N
        )
        self.Beta = beta(
            D_p=self.standard_daily_reward,
            D_m=self.strategy_daily_reward
        )
        self.Alpha = alpha(
            R_p=self.str_total_annualized_returns,
            R_m=self.std_total_annualized_returns,
            B_p=self.Beta
        )
        self.Sigma_p = sigma_p(
            r_p=self.strategy_daily_ratio,
            r__p=np.mean(self.strategy_daily_ratio),
            N=self.N
        )
        self.Sharp_ratio = sharpe_ratio(
            R_p=self.str_total_annualized_returns,
            sigma_p=self.Sigma_p
        )
        self.Max_drawdown = max_drawdown(
            init_money=self.start_money,
            strategy_daily_reward=strategy_daily_reward
        )
        self.kpi_list = [
            self.strategy_daily_reward,
            self.strategy_daily_ratio,
            self.standard_daily_reward,
            self.standard_daily_ratio,
            self.str_money,
            self.std_money,
            self.str_total_returns,
            self.str_total_annualized_returns,
            self.std_total_returns,
            self.std_total_annualized_returns,
            self.Beta,
            self.Alpha,
            self.Sigma_p,
            self.Sharp_ratio,
            self.Max_drawdown
        ]

    def get_str_daily_rewards(self):
        return self.strategy_daily_reward
    def get_str_daily_ratio(self):
        return self.strategy_daily_ratio
    def get_std_daily_rewards(self):
        return self.strategy_daily_reward
    def get_std_daily_ratio(self):
        return self.strategy_daily_ratio
    def get_str_total_returns(self):
        return self.str_total_returns
    def get_str_total_annualized_returns(self):
        return self.str_total_annualized_returns
    def get_std_total_returns(self):
        return self.std_total_returns
    def get_std_total_annualized_returns(self):
        return self.std_total_annualized_returns
    def get_Beta(self):
        return self.Beta
    def get_Alpha(self):
        return self.Alpha
    def get_Sigma_p(self):
        return self.Sigma_p
    def get_Sharpe_ratio(self):
        return self.Sharp_ratio
    def get_Max_drawdown(self):
        return self.Max_drawdown
    def get_kpi(self):
        return self.kpi_list

# if __name__ == '__main__':
#
#     kpi = KPI(
#         data=train_data,
#         paras=paras_list[3],
#         standard_data_file=standard_data_file
#     )
#     # 当前策略
#     print("当前策略:", kpi.strategy)
#     # 策略收益
#     print("策略收益:", kpi.str_money)
#     # 策略收益率
#     print("策略收益率:", kpi.get_str_total_returns())
#     # 策略年化收益率
#     print("策略年化收益率:", kpi.get_str_total_annualized_returns())
#     # 基准收益
#     print("基准收益:", kpi.std_money)
#     # 基准收益率
#     print("基准收益率:", kpi.get_std_total_returns())
#     # 基准年化收益率
#     print("基准年化收益率:", kpi.get_std_total_annualized_returns())
#     # Beta
#     print("Beta:", kpi.get_Beta())
#     # Alpha
#     print("Alpha:", kpi.get_Beta())
#     # 下行波动率
#     print("下行波动率:", kpi.get_Sigma_p())
#     # Sharpe Ratio
#     print("Sharpe Ratio:", kpi.get_Sharpe_ratio())
