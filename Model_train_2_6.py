from Model_init1_LR_2_5 import Model_LR
from Model_init2_BR_2_5 import Model_BR
from Model_init3_SVR_2_5 import Model_SVR
from Model_init4_GBR_2_5 import Model_GBR
from ED_calculate_2_4 import Loc_data
import pandas as pd
import numpy as np

# 模型汇总
Model_all = [Model_LR, Model_BR, Model_SVR, Model_GBR]

# 选择模型为贝叶斯回归【Model_BR】
input_model = Model_BR


# 模型训练及拟合，输入为模型类型和数据，Model_init为【选择的初始模型】，data为输入的【局部数据集】
def model_train(Model_init=input_model, data=Loc_data):
    X = data.iloc[:, :-2]  # X为特征自变量
    Y = data.iloc[:, -2]  # Y为待生定碳因变量
    Model = Model_init.fit(X, Y)  # 模型训练及拟合
    if Model == Model_BR:  # 打印拟合后模型各参数的权重和截距，对应公式的w和b
        print('各特征参数权重：', Model.coef_)
        print('公式截距：', Model.intercept_)
    return Model


Model = model_train(Model_init=input_model, data=Loc_data)
