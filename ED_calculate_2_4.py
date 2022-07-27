from scipy.spatial import distance  # 距离函数 版本：scipy==1.1.0
import pandas as pd
import numpy as np


def ED_calculate(His_path='历史数据库_1800.xlsx', New_path='新样本集_200.xlsx', Loc_path='局部数据集.xlsx'):
    '''
    :param His_path: 历史数据库路径
    :param New_path: 新样本集路径
    :param Loc_path: 局部数据集保存路径
    :return: 【t】计算的第几个样本；【His_data】历史数据集变量；【New_data】新样本集变量；
    【Loc_data】局部数据集变量
    '''
    His_data = pd.read_excel(His_path, index=None)
    New_data = pd.read_excel(New_path, index=None)
    # 数据类型转换
    X_his = np.array(His_data.iloc[:, :-1], dtype=np.float64)
    Y_his = np.array(His_data.iloc[:, -1], dtype=np.float64).reshape(1800, 1)

    X_new = np.array(New_data.iloc[:, :-1], dtype=np.float64)
    Y_new = np.array(New_data.iloc[:, -1], dtype=np.float64).reshape(200, 1)

    window = 50
    # 第一个样本
    t = 0
    His_data['ED'] = 0
    for l in range(1800):
        His_data.iloc[l, -1] = distance.euclidean(X_new[t], X_his[l])
    # 选择前50个样本作为局部数据集，window = 50
    window = 50
    Loc_data = His_data.sort_values(by='ED', ascending=True).iloc[:window, :]  # 按ed从小到达排序并选择前window个样本
    print('局部回归数据集前5行：', Loc_data.head())
    Loc_data.to_excel(Loc_path, index=None)  # 保存局部数据集
    return t, His_data, New_data, Loc_data


t, His_data, New_data, Loc_data = ED_calculate(His_path='历史数据库_1800.xlsx', New_path='新样本集_200.xlsx')
