"""
局部去噪
simple_means_filter() 简单均值滤波
gauss_filter() 高斯滤波
yaroslavsky_filter() 邻域滤波
bilateral_filter() 双边滤波
"""

import numpy as np


def simple_means_filter(n_ima, r_block):
    """
    局部均值滤波\n
    :param n_ima: 待去噪的噪声图像\n
    :param r_block: 图像块半径\n
    :return: 使用均值滤波过后的图像\n
    """
    # Size of the image 获取图像的大小 / 维数
    rows, cols = n_ima.shape

    # 初始化输出图像
    output = np.zeros([rows, cols])

    # Replicate the boundaries of the input image 边界延拓，使得位于图像边缘处的像素点也能形成图像块
    n_ima_pad = np.pad(n_ima, (r_block, r_block), 'symmetric')

    # 权值即当前像素所在块的所有
    # weight = 1 / (r_block * 2 + 1) ** 2

    # 逐行逐列扫描像素进行去噪
    for r in range(rows):
        for c in range(cols):
            # 当前像素的位置
            this_r = r + r_block
            this_c = c + r_block

            # 当前需要去噪的图像所在的图像块
            block = n_ima_pad[r:this_r + r_block + 1, c:this_c + r_block + 1]

            # 对当前图像块使用求平均，等同于每个像素点乘以权值
            output[r, c] = np.mean(block)

    return output


def gauss_filter(n_ima, r_block, k, sigma):
    """
    局部高斯滤波\n
    :param n_ima: 待去噪的噪声图像\n
    :param r_block: 图像块半径\n
    :param k: degree of filtering 常数\n
    :param sigma: noise standard deviation 噪声方差 k * sigma 控制滤波强度\n
    :return: 使用均值滤波过后的图像\n
    """
    # h 滤波器参数，控制滤波强度，值越大，平滑能力越强
    h = (k * sigma) ** 2

    # Size of the image 获取图像的大小 / 维数
    rows, cols = n_ima.shape

    # 初始化输出图像
    output = np.zeros([rows, cols])

    # Replicate the boundaries of the input image
    # 为了处理简单，进行边界延拓，使得位于图像边缘处的像素点也能形成图像块。简化边界处理难度
    n_ima_pad = np.pad(n_ima, (r_block, r_block), 'symmetric')

    # Pre-compute Gaussian distance weights. 计算欧式距离的高斯权重
    n = [i for i in range(-r_block, r_block + 1)]
    [x, y] = np.meshgrid(n, n)
    g_dis_wei = np.exp(-(x ** 2 + y ** 2) / h)

    # 逐行逐列扫描像素进行去噪
    for r in range(rows):
        for c in range(cols):
            # 当前需要去噪的像素点
            this_r = r + r_block
            this_c = c + r_block

            # 当前需要去噪的图像所在的图像块
            block = n_ima_pad[r:this_r + r_block + 1, c:this_c + r_block + 1]

            output[r, c] = sum(sum(np.multiply(g_dis_wei, block))) / sum(sum(g_dis_wei))

    return output


def yaroslavsky_filter(n_ima, r_block, k, sigma):
    """
    局部邻域滤波\n
    :param n_ima: 待去噪的噪声图像\n
    :param r_block: 图像块半径\n
    :param k: degree of filtering 常数\n
    :param sigma: noise standard deviation 噪声方差 k * sigma 控制滤波强度\n
    :return: 使用均值滤波过后的图像\n
    """
    # h 滤波器参数，控制滤波强度，值越大，平滑能力越强
    h = (k * sigma) ** 2

    # Size of the image 获取图像的大小 / 维数
    rows, cols = n_ima.shape

    # 初始化输出图像
    output = np.zeros([rows, cols])

    # Replicate the boundaries of the input image
    # 为了处理简单，进行边界延拓，使得位于图像边缘处的像素点也能形成图像块。简化边界处理难度
    n_ima_pad = np.pad(n_ima, (r_block, r_block), 'symmetric')

    # 逐行逐列扫描像素进行去噪
    for r in range(rows):
        for c in range(cols):
            # 当前需要去噪的像素点坐标
            this_r = r + r_block
            this_c = c + r_block

            # 当前需要去噪的图像所在的图像块
            block = n_ima_pad[r:this_r + r_block + 1, c:this_c + r_block + 1]

            g_int_wei = np.exp(-(block - n_ima_pad[this_r, this_c]) ** 2 / h)

            output[r, c] = sum(sum(np.multiply(g_int_wei, block))) / sum(sum(g_int_wei))

    return output


def bilateral_filter(n_ima, r_block, k, sigma):
    """
    局部双边滤波\n
    :param n_ima: 待去噪的噪声图像\n
    :param r_block: 图像块半径\n
    :param k: degree of filtering 常数\n
    :param sigma: noise standard deviation 噪声方差 k * sigma 控制滤波强度\n
    :return: 使用均值滤波过后的图像\n
    """
    # h 滤波器参数，控制滤波强度，值越大，平滑能力越强
    h = (k * sigma) ** 2

    # Size of the image 获取图像的大小 / 维数
    rows, cols = n_ima.shape

    # 初始化输出图像
    output = np.zeros([rows, cols])

    # Replicate the boundaries of the input image
    # 为了处理简单，进行边界延拓，使得位于图像边缘处的像素点也能形成图像块。简化边界处理难度
    n_ima_pad = np.pad(n_ima, (r_block, r_block), 'symmetric')

    # Pre-compute Gaussian distance weights. 计算欧式距离的高斯权重
    n = [i for i in range(-r_block, r_block + 1)]
    [x, y] = np.meshgrid(n, n)
    g_dis_wei = np.exp(-(x ** 2 + y ** 2) / h)

    # 逐行逐列扫描像素进行去噪
    for r in range(rows):
        for c in range(cols):
            # 当前需要去噪的像素点坐标
            this_r = r + r_block
            this_c = c + r_block

            # 当前需要去噪的图像所在的图像块
            block = n_ima_pad[r:this_r + r_block + 1, c:this_c + r_block + 1]

            # 计算灰度值权重
            g_int_wei = np.exp(-(block - n_ima_pad[this_r, this_c]) ** 2 / h)

            # 最后的权重考虑欧氏距离和灰度值
            weight = g_int_wei * g_dis_wei

            output[r, c] = sum(sum(np.multiply(weight, block))) / sum(sum(weight))

    return output
