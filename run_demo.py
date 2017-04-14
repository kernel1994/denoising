import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from NLmeansfilter import nnl_means_filter
from NLmeansfilter2 import nnl_means_filter2

from LocalFilter import simple_means_filter
from LocalFilter import gauss_filter
from LocalFilter import yaroslavsky_filter
from LocalFilter import bilateral_filter


def calcRMSE(observe, model):
    """
    计算均方根误差
    :param observe: 观察值
    :param model: 实际值
    :return: 均方根误差
    """
    return np.sqrt(np.mean((observe - model) ** 2))


# matlab 文件名
matfn = u'ima.mat'
data = sio.loadmat(matfn)
ima = data['ima'][50:100, 50:100]
nima = np.array(ima)

# ima 的维度
s = nima.shape
# ima 中最大值
maxIma = nima.max()

# Rician 噪声强度
nn = [1, 3, 5, 7, 9]

# 设置非局部均值滤波参数，调用非局部均值去噪函数，进行去噪处理
# radius of search area 非局部均值去噪搜索窗口（邻域窗口）半径大小
r_search = 5
# radius of similarity area 非局部均值去噪图像块半径大小
r_block = 2

noise = []
error_nlmean = []
error_nlmean2 = []
error_lmean = []
error_lgauss = []
error_lyaroslavsky = []
error_lbilateral = []

for i in range(len(nn)):
    # 向实验图像中添加莱斯噪声（Rician Noise）
    level = nn[i] * maxIma / 100
    n1 = level * np.random.random(s)
    n2 = level * np.random.random(s) + nima
    rima = np.sqrt(np.power(n1, 2) + np.power(n2, 2))

    # 使用非局部均值滤波
    # nlmean_ima = nnl_means_filter(rima, r_search, r_block, 1.2, level)
    # 使用带偏差校正的非局部均值滤波
    # nlmean_ima2 = nnl_means_filter2(rima, r_search, r_block, 1.4, level)
    # 使用局部均值滤波
    lmean_ima = simple_means_filter(rima, r_block)
    # 使用局部高斯滤波
    lgauss_ima = gauss_filter(rima, r_block, 1.2, level)
    # 使用局部邻域滤波
    lyaroslavsky_ima = yaroslavsky_filter(rima, r_block, 1.2, level)
    # 使用局部双边滤波
    lbilateral_ima = bilateral_filter(rima, r_block, 1.2, level)

    noise.append(level)
    # 计算去噪后的均方根误差，对去噪结果进行定量比较
    # error_nlmean.append(calcRMSE(nlmean_ima, nima))
    # error_nlmean2.append(calcRMSE(nlmean_ima2, nima))
    error_lmean.append(calcRMSE(lmean_ima, nima))
    error_lgauss.append(calcRMSE(lgauss_ima, nima))
    error_lyaroslavsky.append(calcRMSE(lyaroslavsky_ima, nima))
    error_lbilateral.append(calcRMSE(lbilateral_ima, nima))


plt.figure(1)
# plt.plot(noise, error_nlmean, 'oy-', label="非局部均值滤波")
# plt.plot(noise, error_nlmean2, 'ok-', label="非局部带偏差校正")
plt.plot(noise, error_lmean, 'og-', label="均值滤波")
plt.plot(noise, error_lgauss, 'ob-', label="高斯滤波")
plt.plot(noise, error_lyaroslavsky, 'oc-', label="邻域滤波")
plt.plot(noise, error_lbilateral, 'om-', label="双边滤波")
plt.xlabel('Noise standard deviation')
plt.ylabel('均方根误差')
plt.legend()


def show_plot(origin_ima, noisy_ima, filter_ima, filter_name, i_plot):
    plt.figure(i_plot)

    # plt.figure(2)
    # subplot(rows, cols, num)
    # 整个区域被划分成 rows 行，cols 列，绘制在第 num
    plt.subplot(221)
    plt.imshow(origin_ima, cmap='gray')
    plt.title('原图')

    plt.subplot(222)
    plt.imshow(noisy_ima, cmap='gray')
    plt.title('加噪')

    plt.subplot(223)
    plt.imshow(filter_ima, cmap='gray')
    plt.title(filter_name)

    plt.subplot(224)
    plt.imshow((noisy_ima - filter_ima), cmap='gray')
    plt.title('去掉的噪声')


# show_plot(nima, rima, lmean_ima, '均值滤波', 2)
# show_plot(nima, rima, lgauss_ima, '高斯滤波', 3)
# show_plot(nima, rima, lyaroslavsky_ima, '邻域滤波', 4)
# show_plot(nima, rima, lbilateral_ima, '双边滤波', 5)

# show_plot(nima, rima, nlmean_ima, '非局部均值滤波', 6)

plt.show()
