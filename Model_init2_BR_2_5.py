from sklearn.linear_model import BayesianRidge  # 版本：sklearn==0.21.0

# 初始化贝叶斯回归模型
Model_BR = BayesianRidge(n_iter=300, tol=0.001, fit_intercept=True, normalize=False)

'''
n_iter：int, default=300
最大迭代次数。应大于或等于1。

tol：float, default=1e-3
收敛误差，如果w已收敛，则停止算法。

fit_intercept:bool, default=True
是否计算此模型的截距。截距不作为概率参数，因此没有相关的方差。如果设置为False，
则计算中将不使用截距（即数据应居中）。

normalize:bool, default=False
当fit_intercept设置为False时，忽略此参数。如果为真，则回归前，通过减去平均值并除以l2范数，
对回归数X进行归一化。


原文链接：https://blog.csdn.net/qq_34356768/article/details/107007987
'''
