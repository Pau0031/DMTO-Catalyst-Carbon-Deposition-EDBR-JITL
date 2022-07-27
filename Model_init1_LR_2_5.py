from sklearn.linear_model import LinearRegression  # 版本：sklearn==0.21.0

# 初始化线性回归模型
Model_LR = LinearRegression(fit_intercept=True, normalize=False)

'''
fit_intercept:是否有截据，如果没有则直线过原点。

normalize:是否将数据归一化。
'''

