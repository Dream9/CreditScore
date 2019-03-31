'''
注意，建模数据.csv 已经去除了部分信息(性别、省份等)，同时已经把Y/N替换为了1/0
好坏flag调为0/1
这里主要进行数据的处理工作
包括
1.空缺数据的处理工作，采取了以下策略：缺失占比少的去除此数据行，占比大的去除数据列
2.验证数据之间的相关性，对相关性高的数据项目采取PCA进行降维操作
3.将数据集分割成两部分（测试与训练）,保证随机性


另外，t/f变量没有纳入考虑

'''
import pandas as pd
from pandas import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import logging
import matplotlib
from sklearn.metrics import roc_curve,auc
import pdb
import sys
import json
import math
import copy
from pca import PCA
logging.basicConfig(level=logging.INFO,
                    format='%(lineno)d-%(levelname)s:%(message)s')
logging.getLogger().setLevel(logging.INFO)


rootpath="F:/校赛题目/"
datapath = rootpath+'建模数据.csv'
testpath = rootpath+'testData.csv'
trainpath = rootpath+'TrainData.csv'
newdatapath=rootpath+'newData.csv'
logpath=rootpath+'log.log'
pcatrainpath=rootpath+"train(pca).csv"
pcatestpath=rootpath+"test(pca).csv"
scorepath=rootpath+"score.csv"
alldata_changepath=rootpath+'Data(全部处理过).csv'



'''
@:pram:csv数据
@:return:填充后的csv数据
@:func:采用sklearn的RandomForestRegressor填充数据
'''
def _data_RandomForestRegressor(df):
    process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知特征值和位置特征值两部分
    known = process_df[process_df['MonthlyIncome'].notnull()].as_matrix()
    unknown = process_df[process_df['MonthlyIncome'].isnull()].as_matrix()
    Y = known[:, 0]
    X = known[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, Y)
    predicted = rfr.predict(unknown[:, 1:])
    df.loc[df['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predicted
    return df


'''
@:pram:待处理数据
@:return:去掉均值后的数据和均值
@:func:
'''
def _zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal



'''
@:pram:    data:dataFrame; Y变量：好坏客户，X变量：对X分组，n为组数
@:return:  d4按照'min'排序的DataFrame
           证据权重woe=ln(goodattribute/badattribute) ；
           Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
           四分位数cut 格式为[-inf,a,b,c,d,inf]
@:func:    针对连续变量进行最优分段
'''
def _best_bin(data,Yname, Xname, n):
    print(Xname,"  ",n)
    Y=data[Yname]
    X=data[Xname]
    # 好于坏，，默认好为1，坏为0
    good = Y.sum()
    bad = Y.count() - good
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, n)})
        # try:
        #     d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, n)})
        # except Exception as e:
        #     print(e)
        #     print(Xname,n)
        #     pdb.set_trace()
        # 分组
        d2 = d1.groupby(['Bucket'])
        r, p = stats.spearmanr(d2['X'].mean(), d2['Y'].mean())
        n = n - 1
    # 同时得到了某个d2和n
    print(r,",",n)
    # 统计好坏比率
    d3 = pd.DataFrame(d2['X'].min(), columns=['min'])
    d3['min'] = d2['X'].min()
    d3['max'] = d2['X'].max()
    d3['sum'] = d2['Y'].sum()
    d3['total'] = d2['Y'].count()
    d3['rate'] = d2['Y'].mean()
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    d3['woe'] = -np.log(d3['badattribute']/d3['goodattribute'])
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = d3.sort_index(by='min')
    woe = list(d4['woe'].values)
    logf.write('-------*--------\r\n')

    print(Xname,file=logf)
    print(d4,file=logf)
    logging.info('-' * 30)
    cut = []
    # float('inf') 为正无穷，而不是直接写inf
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    return d4, iv, woe, cut


'''
@:pram:    data:dataFrame; Y变量：好坏客户；X变量：对X分组；cat手动分隔
@:return:  d4按照'min'排序的DataFrame
           证据权重woe=ln(goodattribute/badattribute) ；
           Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
@:func:    针对不能最优分箱的变量进行分段，因此需要手动
'''
def _self_bin(data,Yname, Xname, cat):
    # 好于坏，，默认好为1，坏为0
    Y=data[Yname]
    X=data[Xname]
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.cut(X, cat)})
    d2 = d1.groupby(['Bucket'])
    d3 = pd.DataFrame(d2['X'].min(), columns=['min'])
    d3['min'] = d2['X'].min()
    d3['max'] = d2['X'].max()
    d3['sum'] = d2['Y'].sum()
    d3['total'] = d2['Y'].count()
    d3['rate'] = d2['Y'].mean()
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    d3['woe'] = -np.log(d3['badattribute']/d3['goodattribute'])
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = d3.sort_index(by='min')

    logf.write('-------*--------\r\n')
    logf.flush()
    print(Xname,file=logf)
    print (d4,file=logf)
    woe = list(d3['woe'].values)
    return d4, iv, woe


'''
@:pram:    series:属性；cut：分组；woe权重
@:return:  d4按照'min'排序的DataFrame
           证据权重woe=ln(goodattribute/badattribute) ；
           Infomation Value IV=sum((goodattribute-badattribute)*woe) ;
@:func:    根据woe值反向操作
'''
def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        valuek=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if valuek>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list


'''
@:pram:    list:列属性；data：dataframe; name新插入列的名字
@:return:   pcamain_last：转换矩阵
@:func:    完成pca替换
'''
def pca_merge(pcalist_last,data,name):
    pcadata = data[pcalist_last]
    pcaobj = PCA(pcadata, 1)
    pcaout_last = pcaobj.reduce_dimension()
    pcamatrix = pcaobj.get_feature()
    R = 0
    pcamain_last = pcamatrix.values[0:R + 1, 1:]
    for ii in pcalist_last:
        del data[ii]
    data[name] = pcaout_last
    return  pcamain_last


'''
@:pram:    relate:系数；woe：woe; p:比值因子
@:return:  某忠属性得分
@:func:    计算分数
'''
def get_score(relate,woe,p):
    scores=[]
    for w in woe:
        score=round(relate*w*p,0)
        scores.append(score)
    return scores

'''
@:pram:    series：dataFrame；cut 分隔,scores 某种属性分
@:return:  计算
@:func:    
'''
def compute_score(series,cut,scores):
    i=0
    list=[]
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j=j-1
                m=m-1
        list.append(scores[m])
        i=i+1
    return list




if __name__ == '__main__':
    # 调试参数设置
    is_out_figure = True  # 是否输出cor图像
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    logf = open(logpath, 'w')

    data = pd.read_csv(datapath)
    data = data.iloc[:, :]  #[:,1:]
    data.head()
    # 数据整体预览
    data.describe()
    data.info()
    print(data.columns.to_list())
    print("当前csv属性项",data.columns.to_list(),file=logf)
    '''
    这里已经提前删除了
    缺失较大的数据项：
    flightcount                     1491 non-null float64
    domesticbuscount                1491 non-null float64
    domesticfirstcount              1491 non-null float64
    flightintercount                1491 non-null float64
    avgdomesticdiscount             1491 non-null float64
    缺失较少的
    relevant_stability              20274 non-null float64
    sns_pii                         20261 non-null float64
    '''

    data.dropna(how='any',inplace=True)

    # 单个数据分布
    i5 = 0
    if is_out_figure:
        for ii in data.columns:
            if ii == 'id_rank':
                continue
            i5 += 1
            print(ii)
            mi = data[ii]
            plt.figure()
            plt.title(ii)
            plt.hist(mi, 20)
            plt.savefig(rootpath + ii + "(bin).jpg")
            plt.close()
    # 筛选掉分布极为不均的变量
    # 外部完成







    # 特殊查看
    # 临时使用
    # for ii in data.columns.to_list():
    #     if ii=='credit_score':
    #         continue
    #     print(ii)
    #     plt.figure()
    #     plt.scatter(data[ii],data['credit_score'])
    #     plt.savefig(rootpath+'1111111'+ii+'(1).jpg')
    #     plt.close()
    # pdb.set_trace()


    data2=data.iloc[:,:]
    # 第一列时id,不参与运算
    # 计算变量的相关性
    # 筛选或者pca处理
    cor1 = data2.corr()
    fig = plt.figure()
    # fig.set_size_inches(16, 6)
    ax1 = fig.add_subplot(111)
    sns.heatmap(cor1, vmin=-1, vmax=1, cmap='hsv', square=True)
    # plt.show()
    plt.savefig(rootpath+"cor(1).jpg")
    '''
    初选后的变量部分存在相关性
    需要进一步变换
    credit_pay_amt_1m               
    credit_pay_amt_3m               
    credit_pay_amt_6m               
    credit_pay_amt_1y               
    credit_pay_months_1y            
    credit_total_pay_months 
    

    last_1y_total_active_biz_cnt    
    last_6m_avg_asset_total         
    last_3m_avg_asset_total         
    last_1m_avg_asset_total         
    last_1y_avg_asset_total
    
    tot_pay_amt_6m                  
    tot_pay_amt_3m                 
    tot_pay_amt_1m 

    ebill_pay_amt_6m                
    ebill_pay_amt_3m                
    ebill_pay_amt_1m 
    
    auth_fin_last_1m_cnt            
    auth_fin_last_3m_cnt           
    auth_fin_last_6m_cnt            
    '''
    # 对last部分合并
    pcalist_last=['last_1y_total_active_biz_cnt',
    'last_6m_avg_asset_total',
    'last_3m_avg_asset_total',
    'last_1m_avg_asset_total',
    'last_1y_avg_asset_total']
    pcamain_last=pca_merge(pcalist_last,data,'last_pca')
    # 对credit部分合并
    pcalist_credit = ['credit_pay_amt_1m',
               'credit_pay_amt_3m',
               'credit_pay_amt_6m',
               'credit_pay_amt_1y',
               'credit_pay_months_1y',
               'credit_total_pay_months']
    pcamain_credit = pca_merge(pcalist_credit, data, 'credit_pca')
    # 对tot部分合并
    pcalist_tot = ['tot_pay_amt_6m',
    'tot_pay_amt_3m',
    'tot_pay_amt_1m']
    pcamain_tot = pca_merge(pcalist_tot, data, 'tot_pca')
    # 对ebill部分合并
    pcalist_ebill = ['ebill_pay_amt_6m',
    'ebill_pay_amt_3m',
    'ebill_pay_amt_1m']
    pcamain_ebill = pca_merge(pcalist_ebill, data, 'ebill_pca')
    # 对auth部分合并
    pcalist_ebill = ['auth_fin_last_1m_cnt',
    'auth_fin_last_3m_cnt',
    'auth_fin_last_6m_cnt']
    pcamain_auth = pca_merge(pcalist_ebill, data, 'auth_pca')



    #注意pcamain_credit和pcamain_last负责转换，应该被记录下来
    print("特征矩阵（转换用）",file=logf)
    print('pcamain_credit:\r\n',pcamain_credit,file=logf)
    print('pcamain_last:\r\n', pcamain_last, file=logf)
    print('pcamain_ebill:\r\n', pcamain_ebill, file=logf)
    print('pcamain_auth:\r\n', pcamain_auth, file=logf)
    print('pcamain_tot:\r\n', pcamain_tot, file=logf)


    # 用于判断客户好坏的Y
    Y = data['CREDIT_FLAG']
    X = data.iloc[:, 2:]
    # 更换方式
    datatmp=copy.deepcopy(data)
    del datatmp['CREDIT_FLAG']
    X=datatmp

    print("X.columns.to_list",X.columns.to_list())
    print("Data.columns.to_list",data.columns.to_list())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    # 导出数据
    logging.info("分组导出数据")
    train.to_csv(trainpath, index=False)
    test.to_csv(testpath, index=False)



    # 重新检验相关性
    data.to_csv(rootpath + 'tmp(data-pca).csv',index=False)
    data.describe()
    # 重新检验相关性
    if is_out_figure:
        cor1 = data.corr()
        fig = plt.figure()
        # fig.set_size_inches(16, 6)
        ax1 = fig.add_subplot(111)
        sns.heatmap(cor1, vmin=-1, vmax=1, cmap='hsv', square=True)
        # plt.show()
        plt.savefig(rootpath + "相关性（pca后）.png")


    # 单个数据分布
    '''
    i5 = 0
    for ii in data.columns:
        i5 += 1
        print(ii)
        mi = data[ii]
        plt.figure()
        plt.title(ii)
        plt.hist(mi, 20)
        plt.savefig(rootpath+str(i5)+ii+"(bin).jpg")
        plt.close()
'''







    # pdb.set_trace()
    logging.info("bin")
    # 变量选择
    # 即通过比较指标分箱和对应分箱的违约概率来确定指标是否符合经济意义
    # 首先连续变量最优分段
    # cut是对X取他的四分位，因为Y只有0  1  也不能取四分位。n=3因为最后有n-1，所以实际上是分成了四个桶，woe是四个值。goodattribute是好的属性的意思
    dfx=[]
    ivx=[]
    woe=[]
    cut=[]
    name=[]
    ninf = float('-inf')
    pinf = float('inf')
    i=0
    _N=6
    first=""
    # (Pdb)
    # train.columns.to_list()
    # ['CREDIT_FLAG', 'AGE', 'activity_area_stability', 'adr_stability_days', 'auth_pca', 'consume_steady_byxs_1y',
    #  'credit_duration', 'credit_pca', 'ebill_pca', 'id_rank', 'last_pca', 'positive_biz_cnt_1y', 'pre_1y_pay_amount',
    #  'pre_1y_pay_cnt', 'relevant_stability', 'sns_pii', 'tot_pca', 'use_mobile_2_cnt_1y', 'xfdc_index']
    for ii in train.columns.to_list():
        if ii=='id_rank':
            continue
        name.append(ii)
        if i==0:
            i+=1
            first=ii
            print(ii)
            continue
        if i==1:
            _N=6
        if i==7:
            _N=3
        if i==11:
            _N=5
        if i==13 or i==14:
            _N=10
        if ii=='credit_pca':
            _N=5
        if i<=9 or i==11 or i in [13,14,15,16]:
            a,b,c,d=_best_bin(train,first,ii,_N)
            dfx.append(a)
            ivx.append(b)
            woe.append(c)
            cut.append(d)
            _N=6
        elif i==10:
            cut10 = [ninf, 0,1, 2, 3, pinf]
            a, b, c = _self_bin(train, first, ii,cut10)
            dfx.append(a)
            ivx.append(b)
            woe.append(c)
            cut.append(cut10)
        elif i==12:
            cut12 = [ninf, 1, 2, 3,4, pinf]
            a, b, c = _self_bin(train, first, ii, cut12)
            dfx.append(a)
            ivx.append(b)
            woe.append(c)
            cut.append(cut12)
        elif i==17:
            cut17 = [ninf, -4,-3,-2,-1,pinf]
            a, b, c = _self_bin(train, first, ii, cut17)
            dfx.append(a)
            ivx.append(b)
            woe.append(c)
            cut.append(cut17)
        i+=1

    print("变量name",name,file=logf)
    print('dfx',dfx, file=logf)
    print('ivx值',ivx, file=logf)
    print('cut段',cut, file=logf)

    '''IV筛选'''

    # % matplotlib
    # inline
    ivall = pd.Series(ivx,index=name[1:])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ivall.plot(kind='bar', ax=ax1)
    plt.grid(alpha=0.5)
    # for a, b in zip(indexx,ivlist):
    #     plt.text(a, b+0.01,'%4f' % b,fontsize=10,va='bottom', ha='center')
    # plt.show()
    '''
    根据上一部的计算结果，应该剔除掉<=0.1的值
    '''
    delname=[]
    for ii in range(len(ivx)):
        if ivx[ii]<=0.1:
            del train[name[ii+1]]
            del test[name[ii+1]]
            del data[name[ii+1]]
            delname.append(name[ii+1])
    print('iv不合格，排除的变量',delname,file=logf)

    # 再次导出结果
    train.to_csv(pcatrainpath,index=False)
    test.to_csv(pcatestpath,index=False)
    data.to_csv(alldata_changepath, index=False)
    # 再次验证相关性
    corr = data.corr()
    fig = plt.figure()
    # fig.set_size_inches(16, 6)
    ax1 = fig.add_subplot(111)
    sns.heatmap(corr, vmin=-1, vmax=1, cmap='hsv', square=True)
    # plt.show()







    # woe替换原始数据
    # 使用训练数据
    logging.info("loading （PCA）traindata")
    data = pd.read_csv(pcatrainpath)
    # 顺序是第一种反着来，
    datalist=data.columns.to_list()
    for ii in range(len(woe)):

        if name[ii+1] in delname:
            continue
        if name[ii+1] not in datalist:
            continue
        print(name[ii+1])
        data[name[ii+1]]= Series(replace_woe(data[name[ii+1]],cut[ii],woe[ii]))
    data.to_csv(rootpath+"dddceshi.csv",index=False)
    # Logistic 模型建立
    Y = data['CREDIT_FLAG']
    '''
    data['ovd_order_cNt_3m_m1_status'] = Series(replace_woe(data['ovd_order_cNt_3m_m1_status'], cutx_18, woex_25))
    data['ovd_order_cNt_6m_m1_status'] = Series(replace_woe(data['ovd_order_cNt_6m_m1_status'], cutx_18, woex_26))
    data['ovd_order_cNt_12m_m1_status'] = Series(replace_woe(data['ovd_order_cNt_12m_m1_status'], cutx_18, woex_27))
    data['ovd_order_cNt_12m_m3_status'] = Series(replace_woe(data['ovd_order_cNt_12m_m3_status'], cutx_18, woex_28))
    data['ovd_order_cNt_21_m3_status'] = Series(replace_woe(data['ovd_order_cNt_21_m3_status'], cutx_18, woex_30))
    data['ovd_order_cNt_51_m3_status'] = Series(replace_woe(data['ovd_order_cNt_51_m3_status'], cutx_18, woex_32))'''


    logging.info("loading testdata")
    test = pd.read_csv(pcatestpath)
    # 替换woe，
    datalist = test.columns.to_list()
    for ii in range(len(woe)):

        if name[ii + 1] in delname:
            continue
        if name[ii + 1] not in datalist:
            continue
        print(name[ii + 1])
        test[name[ii + 1]] = Series(replace_woe(test[name[ii + 1]], cut[ii], woe[ii]))
    test.to_csv(rootpath + "tttceshi.csv",index=False)
    # 替换成woe

    # 计算一下相关性：
    cor = data.corr()
    print('最终相关性',cor,file=logf)

    del data['CREDIT_FLAG']
    X = data
    X1 = sm.add_constant(X)
    logit = sm.Logit(Y, X1)
    result = logit.fit()
    print('评价结果：\r\n',result.summary(),file=logf)
    # logf.close()


#     模型验证
    matplotlib.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False
    Y_test = test['CREDIT_FLAG']
    del test['CREDIT_FLAG']
    X_test = test
    logging.info(X_test.shape)
    logging.info(X.shape)
    # 通过ROC曲线和AUC来评估模型的拟合能力。
    X2 = sm.add_constant(X_test)
    resu = result.predict(X2)
    fpr, tpr, threshold = roc_curve(Y_test, resu)

    rocauc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % rocauc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')




    # 模型回归系数
    '''
    adr_stability_days          0.259946            
    pre_1y_pay_cnt              0.147436           
    use_mobile_2_cnt_1y         0.176933            
    activity_area_stability     0.187369            
    positive_biz_cnt_1y         0.048923            
    last_pca                    0.266476            
    auth_pca                    0.386051 
    '''
    coe = [0.259946,
    0.147436,
    0.176933,
    0.187369,
    0.048923,
    0.266476,
    0.386051]
    coe=list(map(lambda x: x*x*10, coe))
    print(coe)


    # p值（比例因子
    _base=550
    p = 20 / math.log(2)
    q = _base - 20 * math.log(20) / math.log(2)
    basescore = round(q + p * coe[0], 0)
    # 因为第一个是常数项
    # 构建评分卡时候只需要选出那些，IV值高的特征就行，最后相加得到总分

    xx=[]
    cnt=0
    # 计算每个部分的评分
    for ii in range(len(woe)):
        if name[ii+1] in delname:
            continue
        try:
            xx.append(get_score(coe[cnt],woe[ii],p))
            cnt+=1
        except Exception as e:
            print(e)
            print(ii)
            print("woe长度",len(woe))
            pdb.set_trace()
    for ii in xx:
        print(ii)

    # x1的四个值分别对应cut的四个区间.PDO Point Double Odds,    就是好坏比翻一倍， odds就是好坏比



    # 对测试数据计算结果
    # 打开需要测试的数据
    calc = pd.read_csv(alldata_changepath)
    # 先构建好Series再加上也可以
    # round可能要用到import math.
    # 只需要对test计算分值，因为我们前面构建模型用的是train，计算分值要用test
    calc['BaseScore'] = Series(np.zeros(len(calc)) + basescore)
    calc['TotalScore']=0+calc['BaseScore']
    cnt=0
    for ii in range(len(woe)):
        if name[ii+1] in delname:
            continue
        calc["Score_"+name[ii]]=Series(compute_score(calc[name[ii+1]], cut[ii],xx[cnt]))
        calc['TotalScore']+= calc["Score_"+name[ii]]
        cnt += 1

    lastscore = calc['TotalScore']
    calc.drop(labels=['TotalScore'], axis=1, inplace=True)
    calc.insert(len(calc.columns), 'TotalScore', lastscore)
    calc.to_csv(scorepath,index=False)


    # 分数分布
    mi = calc["TotalScore"]
    plt.figure()
    plt.title("TotalScore")
    plt.hist(mi, 20)
    plt.savefig(rootpath + "TotalScore_(predict_M1).jpg")


    # 改进：
    # 把最初剔除的变量，选择部分纳入到考量之中

    plt.show()
    logf.close()



