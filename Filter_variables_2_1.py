
import pandas as pd  # 导入pandas 版本：pandas==0.23.0


data = pd.read_excel('样本集_2000.xlsx', index=None)  # 读取数据，用的文件夹下的相对路径，最好绝对路径
print(data.columns)
print(data.head())
'''可选择的变量包括【'外补稀释蒸汽量量', '反应内取保护蒸汽量', '待生汽提段(上部)', '待生汽提段(下部)', '再生剂输送蒸汽量', '浓缩水回炼量',
       '甲醇进料量', '蒸汽配入率', '反应温度', '反应压力', '再生温度', '再生压力', '反应器密相藏量', '再生器密相藏量',
       '反应一旋入口线速', '再生一旋入口线速', '待生定碳', '再生定碳', '循环量', '水洗后二甲醚含量', '水洗后甲醇含量',
       '进入再生器风量', '外补工厂风量风FV-1116', '主风放空量', '主风放空阀位', '主风放空修订', '再生器烧焦总风量',
       '再生滑阀阀位', '双动滑阀A阀位', '双动滑阀B阀位', '生焦率', '乙烯（离线分析数据）', '丙烯（离线分析数据）',
       'C4（离线分析数据）', 'C5及以上（离线分析数据）', '乙烯+丙烯', '乙烯+丙烯+C4', '乙烯+丙烯+C4+C5'】'''
'''1.【'水洗后二甲醚含量', '水洗后甲醇含量'，'乙烯（离线分析数据）', '丙烯（离线分析数据）','C4（离线分析数据）', 'C5及以上（离线分析数据）',
       '乙烯+丙烯', '乙烯+丙烯+C4', '乙烯+丙烯+C4+C5'】均不可作为模型输入变量，原因:离线分析数据,不满足实时性'''
'''2.【'再生剂输送蒸汽量', '浓缩水回炼量', '外补工厂风量风FV-1116', '主风放空修订'】不可作为模型输入变量，原因:记录值均为0，为无效数据'''
'''3.【'循环量'】不可作为模型输入变量，原因：由待生定碳和再生定碳计算而来，不满足可操作性'''


# 点击选择操作，假设选择的参数为【'外补稀释蒸汽量量', '反应内取保护蒸汽量', '待生汽提段(上部)', '待生汽提段(下部)',
#        '甲醇进料量', '蒸汽配入率', '反应温度', '反应压力', '再生温度', '再生压力', '反应器密相藏量', '再生器密相藏量',
#        '反应一旋入口线速', '再生一旋入口线速', '待生定碳', '再生定碳',
#        '进入再生器风量',  '主风放空量', '主风放空阀位',  '再生器烧焦总风量',
#        '再生滑阀阀位', '双动滑阀A阀位', '双动滑阀B阀位', '生焦率'】

input_columns_name = ['外补稀释蒸汽量量', '反应内取保护蒸汽量', '待生汽提段(上部)', '待生汽提段(下部)',
       '甲醇进料量', '蒸汽配入率', '反应温度', '反应压力', '再生温度', '再生压力', '反应器密相藏量', '再生器密相藏量',
       '反应一旋入口线速', '再生一旋入口线速',
       '进入再生器风量',  '主风放空量', '主风放空阀位',  '再生器烧焦总风量',
       '再生滑阀阀位', '双动滑阀A阀位', '双动滑阀B阀位', '生焦率', '待生定碳', '再生定碳']

def data_saixuan(input_columns = input_columns_name, data_before_path = '样本集_2000.xlsx',data_after_path='筛选后数据.xlsx'):
    '''变量筛选函数，【input_columns】为选择的参数名称，
    【data_before_path】为选择前的样本集路径，
    【data_after_path】为经过变量筛选后的数据保存路径
    返回值【data_after】为筛选后的数据变量'''
    data_before = pd.read_excel(data_before_path, index=None)
    data_after = data_before[input_columns]
    print(data_after.head())
    data_after.to_excel(data_after_path, index=None)
    return data_after


data_saixuan(input_columns = input_columns_name, data_before_path = '总数据集.xlsx',
             data_after_path='总数据集_1.xlsx')
