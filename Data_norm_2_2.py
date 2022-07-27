import pandas as pd
from sklearn.preprocessing import MaxAbsScaler  # 导入归一化算法 版本：sklearn==0.21.0
import numpy as np  # numpy库 版本：numpy==1.14.3

def data_norm(Y_name='待生定碳',input_path='筛选后数据.xlsx',save_path= '归一化数据.xlsx'):
    '''数据归一化函数，【Y_name】为预测变量参数，
    【input_path】为第一步筛选变量后的数据存储路径，
    【save_path】为对特征参数归一化后的数据存储路径。
    函数返回值包括【X_norm】特征参数归一化后变量，
    【Y_data】预测参数变量，
    【X_preprocess】特征参数数据预处理公式，
    【Norm_data】归一化后数据变量（包含特征参数和预测值）'''
    data = pd.read_excel(input_path, index=None)
    A = data.copy()
    X_data = A.drop(columns=['待生定碳','再生定碳'])
    Y_data = A[Y_name]
    col_name = X_data.columns
    X_data_1 = np.array(X_data, dtype=np.float64)
    X_preprocess = MaxAbsScaler().fit(X_data_1)
    X_norm = X_preprocess.transform(X_data_1)
    print('展示归一化数据前10个样本：',X_norm[:10,:])
    Norm_data = pd.concat([pd.DataFrame(X_norm,columns=col_name),Y_data],axis=1)
    Norm_data.to_excel(save_path, index=None)
    return X_norm,Y_data,X_preprocess,Norm_data


X_norm,Y_data,X_preprocess,Norm_data = data_norm(Y_name='待生定碳', input_path='总数据集_1.xlsx', save_path='总数据集_2.xlsx')
