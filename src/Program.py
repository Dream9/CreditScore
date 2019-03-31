'''
思路：线性规划问题：


'''
import numpy as np
from scipy import optimize
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
from pca import PCA

rootpath="F:/校赛题目/Method2/"
datapath = rootpath+'建模数据.csv'
testpath = rootpath+'testData.csv'
trainpath = rootpath+'TrainData.csv'
newdatapath=rootpath+'newData.csv'
logpath=rootpath+'log2.log'
pcatrainpath=rootpath+"train(pca).csv"
pcatestpath=rootpath+"test(pca).csv"
scorepath=rootpath+"score.csv"
alldata_changepath=rootpath+'Data(全部处理过).csv'



# 1、python中的scipy中提供了线性规划模块
# # 1）模块导入
# # from scipy import optimize
# # optimize.linprog
# # 2）模块特点说明
# # 需要注意的是这个模块只能找到线性规划的最小值，因此，如果约束条件的不等式（变量除外）中含有“大于等于”的需要修改为“小于等于”，例如：x+y>=5，转换为-x-y<=-5；
# # 最大化max_z=x+y，转换为最小化min_z=-x-y
# # 最后linprog得到的是最小值，添加负号则变为最大值

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
@:pram:    data:dataFrame；name：列名; n分隔数目
@:return:   
@:func:    分段
'''
def segment(data,name,n):
    left=data[name].min()
    right=data[name].max()
    return np.linspace(left,right,n)
'''
@:pram:    
@:return:   
@:func:    找到第一个小于等于的位置  lst升序
'''
def search_lower_bound(lst, key):
    low = 0
    high = len(lst) - 1
    if key <= lst[low]:
        return 0+1
    if key >= lst[high]:
        return high+1
    while low < high:
        mid = int((low + high) / 2)
        if key < lst[mid]:
            high = mid
        else:
            low = mid + 1
    if key <= lst[low]:
        return low+1
'''
@:pram:    
@:return:   
@:func:    计算分数
'''
def score(lst,key,power=1):
    i=search_lower_bound(lst,key)
    return i*power

'''
@:pram:    series：dataFrame；slist: 评分表;px:系数
@:return:  这一列的结果
@:func:    
'''
def compute_score(series,slist,px):
    serieslist=series.to_list()
    i=0
    list=[]
    while i<len(serieslist):
        value=serieslist[i]
        m=score(slist,value)*px
        list.append(m)
        i=i+1
    return list

if __name__ == '__main__':
    # 调试参数设置
    is_out_figure = False  # 是否输出cor图像
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    logf = open(logpath, 'w')

    data = pd.read_csv(datapath)
    data = data.iloc[:, :]  # [:,1:]
    data.head()
    # 数据整体预览
    data.describe()
    data.info()
    '''
    这里已经提前删除了某些列：
    credit_score，PROVINCCE等
    以及缺失较大的数据项：
    flightcount                     1491 non-null float64
    domesticbuscount                1491 non-null float64
    domesticfirstcount              1491 non-null float64
    flightintercount                1491 non-null float64
    avgdomesticdiscount             1491 non-null float64
    
    '''

    data.dropna(how='any', inplace=True)
    # 计算变量的相关性
    # 筛选或者pca处理
    # 去除第一列id
    # del data["id_rank"]
    cor1 = data.corr()
    for index, row in cor1.iterrows():  # 获取每行的index、row
        for col_name in cor1.columns:
            if np.abs(row[col_name])>0.8:
                print("相关性较大：",index,"  ",col_name,row[col_name],"\r\n",file=logf)


    fig = plt.figure()
    # fig.set_size_inches(16, 6)
    ax1 = fig.add_subplot(111)
    sns.heatmap(cor1, vmin=-1, vmax=1, cmap='hsv', square=True)
    plt.savefig(rootpath+"cor(1).png")




    # 分隔数据

    # 单个数据分布
    i5 = 0
    if is_out_figure:
        for ii in data.columns:
            i5 += 1
            print(ii)
            mi = data[ii]
            plt.figure()
            plt.title(ii)
            plt.hist(mi, 20)
            plt.savefig(rootpath+ii+"(bin).jpg")
            plt.close()

    # 外部完成

    # 特殊查看
    # for ii in data.columns.to_list():
    #     if ii=='credit_score':
    #         continue
    #     print(ii)
    #     plt.figure()
    #     plt.scatter(data[ii],data['credit_score'])
    #     plt.savefig(rootpath+'1111111'+ii+'(1).jpg')
    #     plt.close()
    # pdb.set_trace()


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
    
    ovd_order_cnt_1m                
    ovd_order_amt_1m                
    ovd_order_cnt_3m                
    ovd_order_amt_3m                
    ovd_order_cnt_6m               
    ovd_order_amt_6m               
    ovd_order_cnt_12m              
    ovd_order_amt_12m               
    ovd_order_c0t_3m_m1_status     
    ovd_order_c0t_6m_m1_status      
    ovd_order_c0t_12m_m1_status
    
    ovd_order_c0t_12m_m3_status    
    ovd_order_c0t_12m_m6_status    
    ovd_order_c0t_21_m3_status     
    ovd_order_c0t_21_m6_status     
    ovd_order_c0t_51_m3_status   
    ovd_order_c0t_51_m6_status     
    
    pre_1y_pay_cnt               
    pre_1y_pay_amount 
    
    '''
    # 对last部分合并
    pcalist_last = ['last_1y_total_active_biz_cnt',
                    'last_6m_avg_asset_total',
                    'last_3m_avg_asset_total',
                    'last_1m_avg_asset_total',
                    'last_1y_avg_asset_total']
    pcamain_last = pca_merge(pcalist_last, data, 'last_pca')
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
    #对ovd1
    pcalist_ovd1=['ovd_order_cnt_1m',
    'ovd_order_amt_1m',
    'ovd_order_cnt_3m',
    'ovd_order_amt_3m',
    'ovd_order_cnt_6m',
    'ovd_order_amt_6m',
    'ovd_order_cnt_12m',
    'ovd_order_amt_12m',
    'ovd_order_c0t_3m_m1_status',
    'ovd_order_c0t_6m_m1_status',
    'ovd_order_c0t_12m_m1_status']
    pcamain_ovd1 = pca_merge(pcalist_ovd1, data, 'ovd1_pca')
    # 对ovd2
    pcalist_ovd2 = ['ovd_order_c0t_12m_m3_status',
    'ovd_order_c0t_12m_m6_status',
    'ovd_order_c0t_21_m3_status',
    'ovd_order_c0t_21_m6_status',
    'ovd_order_c0t_51_m3_status',
    'ovd_order_c0t_51_m6_status']
    pcamain_ovd2 = pca_merge(pcalist_ovd2, data, 'ovd2_pca')
    # 对pre
    pcalist_pre=['pre_1y_pay_cnt',
    'pre_1y_pay_amount']
    pcamain_pre = pca_merge(pcalist_pre, data, 'pre_pca')

    # 注意pcamain_credit和pcamain_last负责转换，应该被记录下来
    print('pcamain_credit:\r\n', pcamain_credit, file=logf)
    print('pcamain_last:\r\n', pcamain_last, file=logf)
    print('pcamain_ebill:\r\n', pcamain_ebill, file=logf)
    print('pcamain_auth:\r\n', pcamain_auth, file=logf)
    print('pcamain_tot:\r\n', pcamain_tot, file=logf)
    print('pcamain_ovd1:\r\n', pcamain_ovd1, file=logf)
    print('pcamain_ovd2t:\r\n', pcamain_ovd2, file=logf)
    print('pcamain_pre:\r\n', pcamain_pre, file=logf)


    # 再次计算相关性
    cor1 = data.corr()
    fig = plt.figure()
    # fig.set_size_inches(16, 6)
    ax1 = fig.add_subplot(111)
    sns.heatmap(cor1, vmin=-1, vmax=1, cmap='hsv', square=True)
    plt.savefig(rootpath + "cor(2).png")




    # 导出数据
    Y = data['CREDIT_FLAG']
    import copy
    tmpdata=copy.deepcopy(data)
    del tmpdata['CREDIT_FLAG']
    X = tmpdata

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    train.to_csv(trainpath,index=False)
    test.to_csv(testpath,index=False)
    data.to_csv(alldata_changepath,index=False)


#     对train进行训练
    print("train")
    good=train.loc[train['CREDIT_FLAG']==1]
    bad=train.loc[train['CREDIT_FLAG']==-1]
    goodlist = good.columns.to_list()
    badlist=bad.columns.to_list()
    sdict={}
    slist=[]
    # 根据good对每一个分段打分
    _N=8
    name=[]
    # print("goodlist:")
    trainlist=train.columns.to_list()
    for ii in range(len(trainlist)):
        # print(goodlist[ii])
        if trainlist[ii]=='CREDIT_FLAG':
            continue
        if trainlist[ii]=='id_rank':
            continue
        name.append(trainlist[ii])
        slist=segment(train,trainlist[ii],_N)
        sdict[trainlist[ii]]=slist

    print(name)

    _A=600
    _delta=10
    axx=[]
    bx=[]
    zx=[]
    # 目标方程
    for i in range(len(name)):
        zx.append(0)
    zx.append(1)
    Z=np.array(zx)
    # 系数方程
    print("good")
    _C=100
    cnt=0
    for index, row in good.iterrows():  # 获取每行的index、row
        # print(cnt)
        if cnt==_C:
            break
        ax=[]
        for col_name in good.columns:
            if col_name not in name:
                continue
            _s=score(sdict[col_name],row[col_name])
            ax.append(-_s)
        ax.append(1)
        axx.append(ax)
        bx.append(-_A)
        cnt+=1

    print("bad ")
    cnt=0
    for index, row in bad.iterrows():  # 获取每行的index、row
        # print(cnt)
        if cnt==_C:
            break
        ax=[]
        for col_name in bad.columns:
            if col_name not in name:
                continue
            _s=score(sdict[col_name],row[col_name])
            ax.append(_s)
        ax.append(1)
        axx.append(ax)
        bx.append(_A)
        cnt+=1
    bound = (-2000, 2000)
    bounds=(-20,None)
    bound3=(0,40)
    print("Calc")
    print("方程系数：",file=logf)
    print(axx,file=logf)
    print(bx,file=logf)
    print(Z,file=logf)
    logf.close()

    # pdb.set_trace()
    # 限制条件
    line_result = optimize.linprog(Z, A_ub=axx, b_ub=bx,method="interior-point",bounds=(bound,bound,bound,bound,bound,bound,
                                                                                        bound,bound,bound,bound,bound,bound,
                                                                                        bound,bound,bound,bound,bound,bound,
                                                                                        bound,bound,bound3,bound,bounds),
                                   options={"maxiter":10000})#, bounds=(var1_st, var2_st, dd, ss))
    # bound1 = (0, None)
    # bound2 = (None, 0)
    # bound3 = (0, None)
    # bound4 = (0, None)
    # bound5 = (None, 0)
    # bound6 = (None, 0)
    # bound7 = (None, 0)
    # bound8 = (None, 0)
    # bound9 = (0, None)
    # bound10 = (0, None)
    # bound11 = (None, 0)
    # bound12 = (0, None)
    # bound13 = (0, None)
    # bound14 = (None, 0)
    # bound15 = (0, None)
    # bound16 = (0, None)
    # bound17 = (None, 0)
    # bound18 = (0, None)
    # bound19 = (0, None)
    # bound20 = (None, 0)
    # bound21 = (0, None)
    # bound22 = (0, None)
    # bound23 = (None, 0)

    # print("Calc")
    # print("方程系数：", file=logf)
    # print(axx, file=logf)
    # print(bx, file=logf)
    # print(Z, file=logf)
    # logf.close()
    #
    # # pdb.set_trace()
    # # 限制条件
    # line_result = optimize.linprog(Z, A_ub=axx, b_ub=bx, method="interior-point",
    #                                bounds=(bound1, bound2, bound3, bound4, bound5, bound6,
    #                                        bound7, bound8, bound9, bound10, bound11, bound12,
    #                                        bound13, bound14, bound15, bound16, bound17, bound18,
    #                                        bound19, bound20, bound21, bound22, bound23),
    #                                options={"maxiter": 10000})
    # optimize.linprog()
    if line_result.success:
        # 最小值的变量
        print("线性规划最小值的变量：")
        print(line_result.x)
        # 最小值
        print("线性规划最小值：")
        print(line_result.fun)
        # 最大值
        print("线性规划最大值：")
        print(-line_result.fun)
    else:
        print("找不到线性规划最小值。")
        print("Bye")
        exit(0)



    #     计算得分
    px=line_result.x
    # 读取待测试数据
    # print(testpath)
    calc=pd.read_csv(alldata_changepath)
    calc['TotalScore'] = Series(np.zeros(len(calc))+0)
    print(px)
    for ii in range(len(name)):
        print(name[ii])
        # curlist=compute_score(calc[name[ii]],sdict[name[ii]],px[ii])
        # curlist
        # calc["Score_"+name[ii]]=Series(curlist)
        #
        calc["Score_" + name[ii]]=calc[name[ii]].map(lambda x: score(sdict[name[ii]],x)*px[ii])
        calc["TotalScore"]+=calc["Score_"+name[ii]]

    lastscore= calc['TotalScore']
    calc.drop(labels=['TotalScore'], axis=1, inplace=True)
    calc.insert(len(calc.columns),'TotalScore',lastscore)
    calc.to_csv(scorepath,index=False)

    #
    # calc['BaseScore'] = Series(np.zeros(len(calc)) + basescore)
    # calc['TotalScore'] = 0 + calc['BaseScore']
    # cnt = 0
    # for ii in range(len(woe)):
    #     if name[ii + 1] in delname:
    #         continue
    #
    #     calc["Score_" + name[ii][:3]] = Series(compute_score(calc[name[ii + 1]], cut[ii], xx[cnt]))
    #     calc['TotalScore'] += calc["Score_" + name[ii][:3]]
    #     cnt += 1
    #
    # calc.to_csv(scorepath)

    # 计算的分数分布
    mi = calc["TotalScore"]
    plt.figure()
    plt.title("TotalScore")
    plt.hist(mi, 20)
    plt.savefig(rootpath + "TotalScore_(predict_Program).jpg")


    plt.show()


