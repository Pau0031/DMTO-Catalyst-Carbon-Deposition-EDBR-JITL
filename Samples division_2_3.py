from sklearn.model_selection import train_test_split  # 数据划分函数 版本：sklearn==0.21.0
import pandas as pd
import random   # 导入随机包
seed = 1029        # 设定随机数种子
random.seed(seed)

def Data_Split(input_path='归一化数据.xlsx', His_path='历史数据库_1800.xlsx',New_path='新样本集_200.xlsx'):
    '''自定义样本划分函数
    :param input_path: 需要划分数据的路径
    :param His_path: 划分后历史数据库存储路径
    :param New_path: 划分后新样本集存储路径
    :return: 【His_data】历史数据变量，【New_data】新样本数据变量
    '''
    data = pd.read_excel(input_path, index=None)
    His_data, New_data = train_test_split(data, shuffle=True, test_size=0.1463, random_state=11)
    print('历史数据库前五个样本： ', His_data.head())
    print('新样本数据库前五个样本： ', New_data.head())
    His_data.to_excel(His_path, index=None)
    New_data.to_excel(New_path, index=None)
    return His_data, New_data

His_data, New_data = Data_Split(input_path='总数据集_2.xlsx',
        His_path='历史数据库_2000.xlsx',New_path='新样本集_343.xlsx')