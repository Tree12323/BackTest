import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.ticker import FuncFormatter


def main():
    # generate random numbers for temporary test
    n = 100
    month = np.arange(n)
    total_profits = np.random.normal(0, 1, n)     # 累计收益
    daily_profits = np.around(np.random.uniform(-20, 20, n), decimals=2)  # 每日盈亏
    visualize(month, total_profits, daily_profits)


def visualize(month, tp, dp):
    # data processing
    len_tp = len(tp)
    x = range(len_tp)
    _tp = [tp[-1]] * len_tp

    # draw charts
    fig = plt.figure(figsize=(960 / 72, 360 / 72))
    ax = fig.add_subplot(2, 1, 1)

    # line chart of total profits
    ax.plot(x, tp, color='blue')
    line_x = ax.plot(x, _tp, color='skyblue')[0]
    line_tp = ax.axvline(x=len_tp - 1, color='skyblue')

    # represent axis data as percentage
    def to_percent(temp, position):
        return '%1.0f' % (10 * temp) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.axis()
    plt.xticks([])
    # plt.grid(True, linestyle=':', color='orange', alpha=0.6)
    plt.xlabel("时间")
    plt.ylabel("累计收益")
    # data tag
    text0 = plt.text(len_tp - 1, tp[-1], str(tp[-1]), fontsize=10)

    # bar chart of daily profits
    ax = fig.add_subplot(2, 1, 2)
    # generate random numbers for tests
    t1 = np.arange(len(dp))

    # rendering
    tmp = dp.tolist()
    for x in dp:
        num = tmp.index(x)
        if x > 0:
            ax.bar(t1[num], dp[num], color=['r'], edgecolor="white")
        else:
            ax.bar(t1[num], dp[num], color=['g'], edgecolor="white")

    # tag
    # for x,y in zip(t1, dp):
    #     plt.text(x, y, '%.2f' %y + "K", ha="center", va="bottom")

    plt.xlabel("时间")
    plt.ylabel("每日盈亏")
    plt.grid(True, linestyle=':', color='#9999ff', alpha=0.8)
    plt.xticks([])

    # other indexes

    def scroll(event):
        axtemp = event.inaxes
        x_min, x_max = axtemp.get_xlim()
        range_x = (x_max - x_min) / 10
        if event.button == 'up':
            axtemp.set(xlim=(x_min + range_x, x_max - range_x))
        elif event.button == 'down':
            axtemp.set(xlim=(x_min - range_x, x_max + range_x))
        fig.canvas.draw_idle()

    # 这个函数实时更新图片的显示内容
    def motion(event):
        try:
            temp = tp[int(np.round(event.xdata))]
            for i in range(len_tp):
                _tp[i] = temp
            line_x.set_ydata(_tp)
            line_tp.set_xdata(event.xdata)

            text0.set_position((event.xdata, temp))
            text0.set_text(str(temp))

            fig.canvas.draw_idle()  # 绘图动作实时反映在图像上
        except:
            pass

    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('motion_notify_event', motion)

    plt.show()


if __name__ == "__main__":
    main()


