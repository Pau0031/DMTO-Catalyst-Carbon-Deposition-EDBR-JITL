from Model_train_2_6 import Model
from ED_calculate_2_4 import t, Loc_data, New_data, His_data
import pandas as pd


def Predict_Update(His_data, New_data, i=t, path='历史数据库_更新后.xlsx'):
    #  预测
    Pred = Model.predict(New_data.iloc[i, :-1].to_frame().T)
    print('预测值：', Pred)

    #  更新数据库
    His_data.drop(columns='ED', inplace=True)  # 删除新样本集第一个样本计算的欧氏距离列’ED‘
    His_data = pd.concat([His_data, New_data.iloc[i, :].to_frame().T], axis=0)  # 将新数据集第一个样本增加至历史数据集
    print('新数据集前5行：', New_data.head())
    print('历史数据库更新后最后5行：', His_data.tail())
    His_data.to_excel(path, index=None)  # 导出数据库
    return His_data


His_data = Predict_Update(His_data, New_data, i=t, path='历史数据库_更新后.xlsx')
