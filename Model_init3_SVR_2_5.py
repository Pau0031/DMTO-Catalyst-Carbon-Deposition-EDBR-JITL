from sklearn.svm import SVR  # 版本：sklearn==0.21.0

# 初始化支持向量机回归模型
Model_SVR = SVR(kernel='rbf', gamma='auto', tol=0.001)
'''
kernel ： string，optional（default ='rbf'）
指定要在算法中使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。

gamma ： float，optional（default ='auto'）
'rbf'，'poly'和'sigmoid'的核系数。

tol ： float，optional（default = 1e-3）
收敛误差，如果w已收敛，则停止算法。
'''
