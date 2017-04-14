import numpy as np
from LocalFilter import simple_means_filter


def make_kernel(r_block):
    """
    用于计算相似度所需的距离权重函数
    """
    if not r_block:
        return 1

    kernel = np.zeros([2 * r_block + 1, 2 * r_block + 1])

    for d in range(1, r_block + 1):
        value = 1 / (2 * d + 1) ** 2

        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                kernel[r_block - i, r_block - j] = kernel[r_block - i, r_block - j] + value

    return kernel / r_block


def nnl_means_filter2(n_ima, r_search, r_block, k, sigma):
    # Size of the image 获取图像的大小 / 维数
    rows, cols = n_ima.shape

    # Memory for the output 变量定义，用于存放去噪结果图像
    output = np.zeros([rows, cols])
    sweight = np.zeros([rows, cols])

    # Replicate the boundaries of the input image 边界延拓，使得位于图像边缘处的像素点也能形成图像块
    n_ima_pad = np.pad(n_ima, (r_block, r_block), 'symmetric')

    kernel = make_kernel(r_block)
    kernel = kernel / sum(sum(kernel))

    # h 滤波器参数，控制滤波强度，值越大，平滑能力越强
    h = (k * sigma) ** 2
    s2 = 2 * sigma * sigma

    aux = simple_means_filter(n_ima_pad, 1)

    # 逐行、依次对图像中的每个像素点进行非局部均值去噪处理
    for i in range(rows):
        for j in range(cols):
            # 当前需要去噪的点，进行了边界延拓，所以横、纵坐标各要加 r_block
            this_i = i + r_block
            this_j = j + r_block

            # 获取当前像素所在图像块. 即以当前像素为中心，半径为 r_block 的一块
            #  Note: Python 区间为 [m, n)，matlab 中为 [m, n]
            this_block = n_ima_pad[this_i - r_block:this_i + r_block + 1, this_j - r_block:this_j + r_block + 1]

            # 确定当前像素的搜索窗口（或称为邻域窗口）的边界
            # 搜索窗口最小行变量，使搜索窗口始终在图像的原始(未延拓)的第一行及以下
            rmin = max(this_i - r_search, r_block + 1)
            # 搜索窗口最大行变量，使搜索窗口始终在图像的原始(未延拓)的最后行及以上
            rmax = min(this_i + r_search, rows + r_block)
            # 搜索窗口列变量
            smin = max(this_j - r_search, r_block + 1)
            smax = min(this_j + r_search, cols + r_block)

            # 依次计算搜索窗口（或称为邻域窗口）内像素与当前像素的相似性权值
            for r in range(rmin, rmax):
                for s in range(smin, smax):

                    # 不用计算当前像素自己与自己的相似性权值
                    if s <= this_j and r == this_i:
                        continue

                    # TODO: bug
                    if abs(aux[this_i, this_j] - aux[r, s]) > sigma:
                        continue

                    # 获取当前像素的邻域像素所在图像块
                    near_block = n_ima_pad[r - r_block - 1:r + r_block, s - r_block - 1:s + r_block]

                    # 计算图像块之间的欧式距离
                    dd = (this_block - near_block) ** 2
                    # dd = np.multiply((this_block - near_block), (this_block - near_block))
                    d = sum(sum(np.multiply(kernel, dd)))
                    d = d / h

                    # 单减函数，距离越大，函数值越小，计算出当前像素与邻域像素的相似性权值
                    w = 1 / (1 + d * d)

                    sweight[i, j] = sweight[i, j] + w
                    output[i, j] = output[i, j] + w * n_ima_pad[r, s] * n_ima_pad[r, s]

                    sweight[r - r_block, s - r_block] = sweight[r - r_block, s - r_block] + w
                    output[r - r_block, s - r_block] = output[r - r_block, s - r_block] + w * n_ima_pad[this_i, this_j] * n_ima_pad[this_i, this_j]

            sweight[i, j] = sweight[i, j] + 0.5
            output[i, j] = output[i, j] + 0.5 * n_ima[i, j] * n_ima[i, j]

            # Rician correction
            output[i, j] = np.sqrt(max(0, output[i, j] / sweight[i, j] - s2))

    return output
