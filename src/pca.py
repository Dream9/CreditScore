# -*- coding: utf-8 -*-
"""
Created on 2016/11/22
利用Numpy,Pandas和Matplotlib实现PCA,并可视化结果
@author: lguduy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""

    def __init__(self, x, n_components=None):
        """x的数据结构应为ndarray"""
        self.x = x
        self.dimension = x.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)  # 矩阵转秩
        x_cov = np.cov(x_T)  # 协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m, 1)), b))
        c_df = pd.DataFrame(c)
        # print(c_df)
        # 把特征值放在了0处了，不是'0'
        c_df_sort = c_df.sort_values(by=0, axis=0,ascending=False)
        print("当前特征值和特征向量为",c_df_sort)
        # c_df.sort(columns=0, ascending=False)  # 按照特征值大小降序排列特征向量
        return c_df_sort

    def explained_varience_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]

    def paint_varience_(self):
        explained_variance_ = self.explained_varience_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        varience = self.explained_varience_()

        if self.n_components:  # 指定降维维度
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))  # 矩阵叉乘
            return np.transpose(y)
        else:
            print("without n")

digits = datasets.load_digits()
x = digits.data
y = digits.target

if __name__ == '__main__':
    # 测试单元
    pca = PCA(x)
    y = pca.reduce_dimension()

























#
# '''
# 注意，建模数据.csv 已经去除了部分信息(性别、省份等)，同时已经把Y/N替换为了1/0
# 好坏flag调为0/1
# 这里主要进行数据的预处理工作
# 包括
# 1.空缺数据的处理工作，采取了以下策略：缺失占比少的去除此数据行，占比大的去除数据列
# 2.验证数据之间的相关性，对相关性高的数据项目采取PCA进行降维操作
# 3.将数据集分割成两部分（测试与训练）,保证随机性
#
#
# '''
# import pandas as pd
# from pandas import Series
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import scipy.stats as stats
# import statsmodels.api as sm
# import logging
# import matplotlib
# from sklearn.metrics import roc_curve,auc
# import pdb
# import json
# logging.basicConfig(level=logging.INFO,
#                     format='%(lineno)d-%(levelname)s:%(message)s')
# logging.getLogger().setLevel(logging.INFO)
#
#
# datapath = 'F:/校赛题目/建模数据.csv'
# testpath = 'F:/校赛题目/testData.csv'
# trainpath = 'F:/校赛题目/TrainData.csv'
# newdatapath='F:/校赛题目/newData.csv'
# logpath=''
#
# '''
# @:pram:csv数据
# @:return:填充后的csv数据
# @:func:采用sklearn的RandomForestRegressor填充数据
# '''
#
#
# def _data_RandomForestRegressor(df):
#     process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
#     # 分成已知特征值和位置特征值两部分
#     known = process_df[process_df['MonthlyIncome'].notnull()].as_matrix()
#     unknown = process_df[process_df['MonthlyIncome'].isnull()].as_matrix()
#     Y = known[:, 0]
#     X = known[:, 1:]
#     rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
#     rfr.fit(X, Y)
#     predicted = rfr.predict(unknown[:, 1:])
#     df.loc[df['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predicted
#     return df
#
#
# '''
# @:pram:待处理数据
# @:return:去掉均值后的数据和均值
# @:func:
# '''
#
#
# def _zeroMean(dataMat):
#     meanVal = np.mean(dataMat, axis=0)
#     newData = dataMat - meanVal
#     return newData, meanVal
#
#
# '''
# @:pram:数据，前n个成分
# @:return:填充后的csv数据
# @:func:cov函数计算
# '''
# def _pca(dataMat, n):
#     newData, meanVal = _zeroMean(dataMat)
#     covMat = np.cov(newData, rowvar=0)
#     eigVals, eigVects = np.linalg.eig(np.mat(covMat))
#     eigValIndice = np.argsort(eigVals)
#     n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
#
#     n_eigVect = eigVects[:, n_eigValIndice]
#     lowDDataMat = newData * n_eigVect
#     reconMat = (lowDDataMat * n_eigVect.T) + meanVal
#     return lowDDataMat, reconMat
#
#
#
# '''
# @:pram:    Y变量：好坏客户，X变量：对X分组，n为组数
# @:return:  d4按照'min'排序的DataFrame
#            证据权重woe=ln(goodattribute/badattribute) ；
#            Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
#            四分位数cut 格式为[-inf,a,b,c,d,inf]
# @:func:    针对连续变量进行最优分段
# '''
# def _best_bin(Y, X, n):
#     # 好于坏，，默认好为1，坏为0
#     good = Y.sum()
#     bad = Y.count() - good
#     r = 0
#     while np.abs(r) < 1:
#         d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, n)})
#         # 分组
#         d2 = d1.groupby(['Bucket'])
#         r, p = stats.spearmanr(d2['X'].mean(), d2['Y'].mean())
#         n = n - 1
#     # 同时得到了某个d2和n
#     print(r,",",n)
#     # 统计好坏比率
#     d3 = pd.DataFrame(d2['X'].min(), columns=['min'])
#     d3['min'] = d2['X'].min()
#     d3['max'] = d2['X'].max()
#     d3['sum'] = d2['Y'].sum()
#     d3['total'] = d2['Y'].count()
#     d3['rate'] = d2['Y'].mean()
#     d3['goodattribute'] = d3['sum'] / good
#     d3['badattribute'] = (d3['total'] - d3['sum']) / bad
#     d3['woe'] = np.log(d3['goodattribute'] / d3['badattribute'])
#     iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
#     d4 = d3.sort_index(by='min')
#     woe = list(d4['woe'].values)
#     print(d4)
#     logging.info('-' * 30)
#     cut = []
#     # float('inf') 为正无穷，而不是直接写inf
#     cut.append(float('-inf'))
#     for i in range(1, n + 1):
#         qua = X.quantile(i / (n + 1))
#         cut.append(round(qua, 4))
#     cut.append(float('inf'))
#     return d4, iv, woe, cut
#
#
# '''
# @:pram:    Y变量：好坏客户；X变量：对X分组；cat手动分隔
# @:return:  d4按照'min'排序的DataFrame
#            证据权重woe=ln(goodattribute/badattribute) ；
#            Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
# @:func:    针对不能最优分箱的变量进行分段，因此需要手动
# '''
# def _self_bin(Y, X, cat):
#     # 好于坏，，默认好为1，坏为0
#     good = Y.sum()
#     bad = Y.count() - good
#     d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.cut(X, cat)})
#     d2 = d1.groupby(['Bucket'])
#     d3 = pd.DataFrame(d2['X'].min(), columns=['min'])
#     d3['min'] = d2['X'].min()
#     d3['max'] = d2['X'].max()
#     d3['sum'] = d2['Y'].sum()
#     d3['total'] = d2['Y'].count()
#     d3['rate'] = d2['Y'].mean()
#     d3['goodattribute'] = d3['sum'] / good
#     d3['badattribute'] = (d3['total'] - d3['sum']) / bad
#     d3['woe'] = np.log(d3['goodattribute'] / d3['badattribute'])
#     iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
#     d4 = d3.sort_index(by='min')
#     print(d4)
#     woe = list(d3['woe'].values)
#     return d4, iv, woe
#
#
# '''
# @:pram:    series:属性；cut：分组；woe权重
# @:return:  d4按照'min'排序的DataFrame
#            证据权重woe=ln(goodattribute/badattribute) ；
#            Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
# @:func:    根据woe值反向操作
# '''
# def replace_woe(series,cut,woe):
#     list=[]
#     i=0
#     while i<len(series):
#         valuek=series[i]
#         j=len(cut)-2
#         m=len(cut)-2
#         while j>=0:
#             if valuek>=cut[j]:
#                 j=-1
#             else:
#                 j -=1
#                 m -= 1
#         list.append(woe[m])
#         i += 1
#     return list
#
#
# if __name__ == '__main__':
#     data = pd.read_csv(datapath)
#     data = data.iloc[:, :]  #[:,1:]
#     data.head()
#     # 数据整体预览
#     data.describe()
#     data.info()
#     '''
#     缺失较大的数据项：
#     flightcount                     1491 non-null float64
#     domesticbuscount                1491 non-null float64
#     domesticfirstcount              1491 non-null float64
#     flightintercount                1491 non-null float64
#     avgdomesticdiscount             1491 non-null float64
#     缺失较少的
#     relevant_stability              20274 non-null float64
#     sns_pii                         20261 non-null float64
#     '''
#     # _data_RandomForestRegressor(data)
#     data.dropna(how='any',inplace=True)
#     # pca变换
#     # olddata=data.iloc[:,-11:-2]
#     # newdata=pca(olddata,1)
#
#     # 分隔数据
#
#     # 用于判断客户好坏的Y
#     Y = data['CREDIT_FLAG']
#     X = data.iloc[:, 1:]
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#
#     train = pd.concat([Y_train, X_train], axis=1)
#     test = pd.concat([Y_test, X_test], axis=1)
#
#     # train.to_csv(trainpath, index=False)
#     # test.to_csv(testpath, index=False)
#     i5=0
#     for ii in data.columns:
#         i5+=1
#         print(ii)
#         mi=data[ii]
#         plt.figure()
#         plt.title(ii)
#         plt.hist(mi,10)
#         plt.savefig("D:/abc%d.jpg" % i5)
#         plt.close()