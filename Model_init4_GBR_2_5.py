from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法 版本：sklearn==0.21.0

#  初始化梯度提升回归模型
Model_GBR = GradientBoostingRegressor(loss='ls', n_estimators=220, learning_rate=0.1, max_depth=3)
'''
loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)
# 算法中选用的损失函数
# 对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”
# 默认是均方差"ls"
# 一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好
# 如果是噪音点较多，推荐用抗噪音的损失函数"huber"
# 而如果我们需要对训练集进行分段预测的时候，则采用“quantile”

 
learning_rate : float, optional (default=0.1)
# 即每个弱学习器的权重缩减系数a，也称作步长,# a的取值范围为[0,1] 
# 对于同样的训练集拟合效果，较小的a意味着需要更多弱学习器迭代次数
# 一般来说这个值不应该设的比较大，因为较小的learning rate使得模型对不同的树更加稳健，
# 就能更好地综合它们的结果
# 是为了防止过拟合而加上的正则化项系数


n_estimators : int (default=100) 
# 弱学习器的个数，或者弱学习器的最大迭代次数
# 一般来说 n_estimators 太小，容易欠拟合；n_estimators 太大，容易过拟合


max_depth : integer, optional (default=3)
# 定义了树的最大深度。
# 它也可以控制过度拟合，因为分类树越深就越可能过度拟合
# 一般来说，数据少或者特征少的时候可以不管这个值
# 如果模型样本量多，特征也多的情况下，推荐限制这个最大深度
# 具体的取值取决于数据的分布，常用的可以取值 10-100 之间
原文链接：https://blog.csdn.net/zhsworld/article/details/102951061
'''
