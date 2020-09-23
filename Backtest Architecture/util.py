import numpy as np
from scipy import interpolate

def calval(a, b):
    means = (a + b) / 2
    sigma = abs(a - b) / 6
    s = np.random.normal(means, sigma)
    return s


def interpolation_single_fea(yp, n=2):
    yp.tolist()
    y = []

    # 两点取平均
    if (n == 2):
        # y轴取均值插入
        for i in range(0, len(yp) - 1):
            y.append(yp[i])
            y.append((yp[i] + yp[i + 1]) / 2)
        y.append(yp[-1])

    # 四点取平均
    elif (n == 4):
        # 四个点y值取平均
        y.append(yp[0])
        for i in range(1, len(yp) - 2):
            y.append(yp[i])
            y.append((yp[i - 1] + yp[i] + yp[i + 1] + yp[i + 2]) / 4)
        y.append(yp[-2])
        y.append(yp[-1])

    # 每一个插值点由某一个高斯分布采样产生
    elif (n == 0):
        for i in range(0, len(yp) - 1):
            y.append(yp[i])
            s = calval(yp[i], yp[i + 1])
            y.append(s)
        y.append(yp[-1])

    # 每一个插值点由某一个高斯分布采样产生，再对真实点增加高斯噪声
    elif (n == 1):
        for i in range(0, len(yp) - 1):
            s = calval(yp[i], yp[i + 1])
            if i != 0:
                y.append(np.random.normal(yp[i], min(abs(yp[i - 1] - yp[i]), abs(yp[i] - s)) / 12))
            else:
                y.append(yp[i])
            y.append(s)
        y.append(yp[-1])

    # 三次样条插值
    elif (n == 3):
        y.append(yp[0])
        f = interpolate.interp1d(np.array((0, 2, 4, 6)), yp[:4], kind="cubic")
        y.append(f(1))
        y.append(yp[1])
        y.append(f(3))
        for i in range(2, len(yp) - 2):
            y.append(yp[i])
            f = interpolate.interp1d(np.array((0,2,4,6)), yp[i-1:i+3], kind="cubic")
            y.append(f(3))
            # y.append(np.random.normal(f(3), min(abs(yp[i] - f(3)),abs(f(3) - yp[i+1]) / 6)))
        y.append(yp[-2])
        y.append(f(5))
        y.append(yp[-1])

    elif (n == 5):
        y.append(yp[0])
        f = interpolate.interp1d(np.array((0,2,4,6)), yp[0:4], kind="quadratic")
        y.append(f(1))
        y.append(yp[1])
        y.append(f(3))
        for i in range(2, len(yp) - 2):
            y.append(yp[i])
            f = interpolate.interp1d(np.array((0,2,4,6)), yp[i-1:i+3], kind="quadratic")
            y.append(f(3))
            # y.append(np.random.normal(f(3), min(abs(yp[i] - f(3)),abs(f(3) - yp[i+1]) / 6)))
        y.append(yp[-2])
        y.append(f(5))
        y.append(yp[-1])

    # 返回值：插值扩充之后的时间序列[x, y]
    return y


def interpolation(yp, n=2):
    fea_num = yp.shape[1]
    y = []
    for i in range(fea_num):
        y.append(interpolation_single_fea(yp[:, i], n))

    y = np.array(y).T

    return y


