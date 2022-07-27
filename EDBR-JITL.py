from scipy.spatial import distance
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import  mean_absolute_error,\
    mean_squared_error, r2_score,mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import time
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import os
seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

# 数据加载
def data_pre(path):
    data20 = pd.read_excel(path, index=None)
    Data0 = np.array(data20, dtype=np.float64) # 转换为ndarray格式

    X_data = Data0[:, 0:22]

    Y_data = Data0[:, 22:24]


    # max_abs = MaxAbsScaler() # 定义数据归一化
    # X_norm = max_abs.fit_transform(X_data)

    X_process = MinMaxScaler().fit(X_data)  # 数据归一化
    X_norm = X_process.transform(X_data)

    # DB=database
    # 划分数据集
    x_DB, x_new, y_DB, y_new = train_test_split(X_norm, Y_data[:, 0],
                                                shuffle=True, test_size=0.1463, random_state=11)

    y_DB = y_DB.reshape(len(y_DB),1)
    y_new = y_new.reshape(len(y_new),1)
    return x_DB,y_DB,x_new,y_new,data20,X_process

# 模型训练预测及更新
def model_initial_train_predict_updata(x_new,y_new,x_DB,y_DB,window):
    model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
    br = []
    col = x_new.shape[1]
    #window = 70 # 回归数据集样本数大小
    for n in range(0,len(x_new[:,0])):
        dist = []
        for i in range(0,len(x_DB[:,0])):
            dist.append(distance.euclidean(x_new[n],x_DB[i]))  # 计算欧式距离
        D0 = np.array(dist, dtype=np.float64)
        D1 = D0.reshape(len(x_DB[:, 0]), 1)
        x_DB1 = np.append(x_DB, D1, axis=1)
        x_DB1 = np.append(x_DB1, y_DB, axis=1)
        n_DB = x_DB1[np.argsort(x_DB1[:, col])]
        data_r = n_DB[0:window, :]  # 筛选前50个相关样本作为回归数据集
        x_r = data_r[:, :col]
        y_r = data_r[:, -1]
        # 估计
        br.append(model_br.fit(x_r,y_r).predict(x_new[n].reshape(1, col))) # 模型拟合及预测
        #间隔20个样本打印一次权重及截距
        # if n in range(0,239,20):
        #     print("第%s个新样本权重分布：" % (n), model_br.coef_)
        #     print("第%s个新样本截距：" % (n), model_br.intercept_)
        #     print(100 * '-')  # 打印分隔线
        # 更新数据库
        x_DB = np.append(x_DB, x_new[n].reshape(1, col),axis=0)
        y_DB = np.append(y_DB, y_new[n].reshape(1, 1),axis=0)
    br_y = np.array(br)
    return br_y,model_br,window


def max_error(y_true, y_predict):  # 定义最大绝对误差
    error2 = np.max(np.abs(y_true - y_predict), axis=0).round(5)
    return error2

# 计算误差
def cal_error(y_new, y_pre):
    y_name = [y_pre]
    model_metrics_name = [mean_absolute_error, mean_squared_error,
                          r2_score, mean_absolute_percentage_error,max_error]  # 回归评估指标对象集, max_error, 'Max AE'
    model_metrics_list = []  # 回归评估指标列表
    for i in y_name:  # 循环每个模型索引
        tmp_list = []
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_new, i)  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

    all_error = pd.DataFrame(model_metrics_list, index=["BR"],
                             columns=['MAE', 'MSE', 'r2', 'MAPE','mAE'])
    print(100 * '-')  # 打印分隔线
    print(all_error)
    print(100 * '-')  # 打印分隔线
    return all_error

# 绘制真实值、预测值折线图
# 结果可视化
def visualization(y_true, y_predict):
    plt.figure(1)  # 创建画布
    plt.plot(np.arange(50), y_true[290:340], color='r', marker='o', label='真实值')  # 画出原始值new_data.shape[0]的曲线
    # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(50), y_predict[290:340], c='blue',
             marker='x', label='预测值')  # 画出每条预测结果线
    plt.title('18-21年待生定碳回归EDBR-JITL(windows=%d)' % (wd))  # 标题
    plt.legend(loc='lower right')  # 图例位置
    plt.ylabel('催化剂有机物质量百分比/%')  # y轴标题
    plt.show()  # 展示图像
    # print("打印模型估计值如下", y_predict)

wd = 100

x_DB, y_DB, x_new, y_new, data20, X_process = data_pre('总数据集_1.xlsx')
T1 = time.time()
y_pre, model_br, window = model_initial_train_predict_updata(x_new, y_new, x_DB ,y_DB,wd)
print("window = ", wd)
all_error = cal_error(y_new,y_pre)
visualization(y_new, y_pre)
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
np.set_printoptions(precision=5)



