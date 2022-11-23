import concurrent

import joblib
# #import os
# #os.environ['TF_KERAS']='1'  # when tensorflow<2.0
# from keras_self_attention import SeqSelfAttention

# # from keras_self_attention import SeqSelfAttention
# from hyperas import optim
# from hyperas.distributions import choice, uniform
# from hyperopt import Trials, STATUS_OK, tpe

import pandas as pd;pd.set_option('mode.chained_assignment', None)
import numpy as np
from numpy import *

from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,matthews_corrcoef,precision_recall_curve,confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import math
from itertools import repeat
import difflib
from tqdm import tqdm
import gc


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn    import  svm
import lightgbm as lgb
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn import svm
from sklearn import neighbors
from numpy import *
import seaborn as sns


def cum_95CI(j):
    for i in range(10000):
        # bootstrap by sampling with replacement on the prediction indices
        indices=np.random.randint(0,len(pro_comm_Pre) - 1,int(len(pro_comm_Pre)/2))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        else:
            break
    
    #    score = roc_auc_score(y_true[indices], y_pred[indices])
    eva_CI = evaluating_indicator(y_true=y_true[indices], y_test=blo_comm_Pre[indices], y_test_value=pro_comm_Pre[indices])
    return pd.DataFrame(eva_CI,index=[0])
    
def cumCI(y_test):
    global y_true
    global pro_comm_Pre
    y_true=np.array(y_test)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # pro=list(executor.map(calculate_sampling_frequency,icustay_id))
        pro=list(tqdm(executor.map(cum_95CI,range(0,10000)),total=10000,desc= ' cum_95CI Processing'))
        # res=list(tqdm(p.imap(function,params),total=len(params),desc='Processing'))
    input_mulit=('pro[{}]'.format(0))
    for i in range(1,len(pro)):
        input_mulit=(input_mulit+',pro[{}]'.format(i))
    cum_95CI_pro=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    
    for i in ['ACC','AUC','BER','F1_score','KAPPA','MCC','SEN','SPE']:   
        sorted_scores = np.array(cum_95CI_pro[i]); sorted_scores.sort()
        print("Confidence interval for the "+i+": [{:0.6f} - {:0.6}]".format(sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]))
    return



def kappa(matrix):   #增加对 评价指标：KAPPA系数的计算
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)


def evaluating_indicator(y_true, y_test, y_test_value):  #计算预测结果的各项指标
    c_m = confusion_matrix(y_true, y_test)
    TP=c_m[0,0]
    FN=c_m[0,1]
    FP=c_m[1,0]
    TN=c_m[1,1]
    
    SEN=TP/ (TP+ FN) #灵敏性
    SPE= TN / (FP + TN) #特异性
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_true, y_test)
    MCC = matthews_corrcoef(y_true, y_test)
    F1score =  f1_score(y_true, y_test)
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    
    # precision , recall,thresholds = precision_recall_curve(y_true,y_test_value)
    # AUPRC=(precision[1:]*(recall[:-1]-recall[1:])).sum()
    KAPPA=kappa(c_m)
    from tqdm import tqdm
    c={"SEN" : SEN,"SPE" : SPE,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC,'KAPPA':KAPPA
    }  #以字典的形式保存预测结果
    return c

def blo(pro_comm_Pre,jj):     #根据预测概率与最优分类阈值对患者进行二类别预测
    blo_Pre=zeros(len(pro_comm_Pre)).reshape(-1,1)
    blo_Pre[(pro_comm_Pre[:,1]>(jj*0.01))]=1
    return blo_Pre



def inter_learn_len(chart_time_opt,icu_id_param,label_name):
    icu_id_param=icu_id_param[icu_id_param.chart_time<=chart_time_opt[-1]]
    if icu_id_param[icu_id_param.chart_time==0].shape[0]==0:
        icu_id_param=pd.concat([icu_id_param[icu_id_param.chart_time==list(set(icu_id_param.chart_time))[0]],icu_id_param],ignore_index=True)
        icu_id_param.chart_time.iloc[0]=0;icu_id_param[label_name].iloc[0]=np.nan

    for i in range(0,int(chart_time_opt[-1])):
        if icu_id_param[icu_id_param.chart_time==i].shape[0]==0:
            icu_id_param=pd.concat([icu_id_param[icu_id_param.chart_time<i],icu_id_param[icu_id_param.chart_time<i].iloc[-1].to_frame().T,icu_id_param[icu_id_param.chart_time>i]],ignore_index=True)
            icu_id_param.chart_time.iloc[i]=i;icu_id_param[label_name].iloc[i]=np.nan
    return icu_id_param



def reconstruction_lstm(icustay_id):
    is_train=1
    icu_id_param = comtest[comtest.icustay_id==icustay_id]   # 选择待重构患者数据
    icu_id_param = icu_id_param.groupby(['chart_time'],as_index=False).max()
    
    chart_time_opt=np.array(icu_id_param.chart_time[((np.array(icu_id_param[labelname])==1) | (np.array(icu_id_param[labelname])==0)).reshape(-1)])
    temp_param_return = pd.DataFrame(np.array([[] for i in range(len(name_param))]).T,columns=name_param)
    temp_label_return = pd.DataFrame(np.array([[] for i in range(fore_len*len(name_label))]).T,columns=new_name)
    # for_test = pd.DataFrame(np.array([]).T,columns=['for_test'])
    if len(chart_time_opt)!=0:
        icu_id_param=inter_learn_len(chart_time_opt,icu_id_param,labelname)

        for jj in chart_time_opt:
            if jj-gap-learn_len>=0 and jj+fore_len<=icu_id_param.shape[0]:  #此时jj为预测步长第一个
                if (is_train!=1) & (np.array((icu_id_param.loc[jj-gap-learn_len:jj-1,labelname].sum()>0)+0)[0]==1):
                    continue
                temporary_param=icu_id_param[(icu_id_param.chart_time<jj-gap)&(icu_id_param.chart_time>=jj-gap-learn_len)][name_param]
                temporary_label=icu_id_param[(icu_id_param.chart_time<jj+fore_len)&(icu_id_param.chart_time>=jj)][name_label]
                templabel=np.array([]).reshape(1,-1)
                for i in name_label:
                    templabel=np.hstack((templabel,np.array(temporary_label[i]).reshape(1,-1)))
                temporary_label=pd.DataFrame(templabel,columns=new_name)
                
                temp_param_return=pd.concat([temp_param_return,temporary_param],axis=0)
                temp_label_return=pd.concat([temp_label_return,temporary_label],axis=0)
    return temp_param_return,temp_label_return


def reconstructe_for_modeling(pro,work_for_conformal):
    input_mulit=('pro[{}][0]'.format(0))
    for i in range(1,len(pro)):
        input_mulit=(input_mulit+',pro[{}][0]'.format(i))
    param=pd.concat(eval(input_mulit),axis=0,ignore_index=True);
    if work_for_conformal==0:
        param=param.iloc[:,1:param.shape[1]]
     
    input_mulit=('pro[{}][1]'.format(0))
    for i in range(1,len(pro)):
        input_mulit=(input_mulit+',pro[{}][1]'.format(i))
    label=pd.concat(eval(input_mulit),axis=0,ignore_index=True);del input_mulit;gc.collect()
    
    for i in range(1,fore_len):
        label.drop('icustay_id_'+str(i),axis=1,inplace=True)
    label.rename(columns = {'icustay_id_'+str(fore_len): 'icustay_id'},inplace=True)
    label=label.iloc[:,1:]
    return param,label

gap= 1*2
fore_len= 1*2 # fore_len
fore_len_in_labelcsv=fore_len*1  #csv 文件中实际的预测步长 
# batch_size = 32*80*2
timesteps =1 # learn
learn_len=timesteps
labelname=['mods_label']

global comtest;global name_param;global name_label;global new_name
database = 'mimic4'

#######SQL_all###############################################
# param_name=['icustay_id','chart_time','age', 'gender', 'fio2', 'spo2', 'temperature', 'hr', 'rr', 'sbp', 'dbp',
#                 'dobutamine', 'dopamine', 'epinephrine','norepinephrine', 'pf_ratio', 'platelet', 'bilirubin',
#                 'mbp', 'gcs', 'gcs_v', 'gcs_e', 'gcs_m', 'creatinine', 'uo', 'uosum',
#                 'hematocrit', 'potassium', 'sodium', 'ph', 'urea', 'wbc', 'pco2', 'cal_co2', 'po2', 'base_excess', 'chloride',
#                 'glucose', 'bicarbonate', 'magnesium', 'phosphate', 'calcium_t', 'pt', 'free_calcium', 'lactate', 'alt', 'ast',
#                 'sofa_sum', 'qsofa_sum', 'lod_sum', 'ma_sum', 'vent_label', 'death_label', 'mods_label']




# #####youchuang##############################
# param_name=['icustay_id','chart_time','age', 'gender', 'fio2', 'spo2', 'temperature', 'hr', 'rr', 'sbp', 'dbp',
#                 'dobutamine', 'dopamine', 'epinephrine','norepinephrine', 'pf_ratio', 'platelet', 'bilirubin',
#                 'mbp', 'gcs', 'gcs_v', 'gcs_e', 'gcs_m', 'creatinine', 'uo', 'uosum',
#                 'hematocrit', 'potassium', 'sodium', 'ph', 'urea', 'wbc', 'pco2', 'cal_co2', 'po2', 'base_excess', 'chloride',
#                 'glucose', 'bicarbonate', 'magnesium', 'phosphate', 'calcium_t', 'pt', 'free_calcium', 'lactate', 'alt', 'ast', 'vent_label', 'mods_label']

#
###################wuchang######################################
#param_name=['icustay_id','chart_time','age', 'gender', 'fio2', 'spo2', 'temperature', 'hr', 'rr', 'sbp', 'dbp',
#                'dobutamine', 'dopamine', 'epinephrine','norepinephrine',
#                'mbp', 'gcs', 'gcs_v', 'gcs_e', 'gcs_m', 'uo', 'uosum', 'vent_label', 'mods_label']



###################wuren######################################
param_name=['icustay_id','chart_time','age', 'gender', 'fio2', 'spo2', 'temperature', 'hr', 'rr', 'sbp', 'dbp','mbp','vent_label','mods_label']

comtest_train = pd.read_csv('/home/guanjun/Liu/dabiao_1113.csv',usecols=param_name)[param_name];icustay_id = set(comtest_train.icustay_id)
scaler = StandardScaler()   #对病例数据进行标准化处理


comtest_train.iloc[:,4:-1] = scaler.fit_transform(comtest_train.iloc[:,4:-1])

x_train_id, x_test_id, y_train_id, y_test_id = model_selection.train_test_split(np.array(list(icustay_id)),range(len(icustay_id)), test_size = 0.2,random_state = 34) 
    # x_train_id, x_val_id, y_train_id, y_val_id = model_selection.train_test_split(np.array(x_train_id),range(len(x_train_id)), test_size = 0.2,random_state = 1) 

name_param=list(comtest_train)[:-1]
    # name_label=['chart_time','hemo_shock_label','shock_label','death_label']
name_label=['icustay_id','mods_label']
new_name=[]
for j in name_label:
        for i in range(1,fore_len+1):
            new_name=new_name+[j+'_'+str(i)]

comtest=comtest_train.loc[comtest_train['icustay_id'].isin(x_train_id)];icustay_id = x_train_id
with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(tqdm(executor.map(reconstruction_lstm,icustay_id),total=len(icustay_id),desc=database+' trainset reconstruction Processing'))
x_train,y_train = reconstructe_for_modeling(pro,0);#x_train.drop(['chart_time'],axis=1,inplace = True);
del pro;gc.collect()
    
comtest=comtest_train.loc[comtest_train['icustay_id'].isin(x_test_id)];icustay_id = x_test_id
with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(tqdm(executor.map(reconstruction_lstm,icustay_id),total=len(icustay_id),desc=database+' test reconstruction Processing'))
x_test,y_true = reconstructe_for_modeling(pro,0);#x_test.drop(['chart_time'],axis=1,inplace = True);
del pro;gc.collect()

y_train=y_train.max(axis=1)
y_true_x =y_true.max(axis=1)


comm = lgb.LGBMClassifier()#(num_leaves=16,max_depth = 8,learning_rate=0.07,n_estimators=150,subsample=0.8,reg_alpha=0.1,reg_lambda=1)
# print('mods')
# #comm = MLPClassifier()
# comm = XGBR()
# comm = RandomForestClassifier()
# comm = LogisticRegression()
#comm = lgb.LGBMClassifier()
# print('svm pro ')
# comm = svm.SVC(probability=True)
# # comm = ensemble.AdaBoostClassifier(learning_rate =0.1, n_estimators=500)
# a=0.2
# comm = GNB=GaussianNB(priors=[a,1-a])
# comm = neighbors.KNeighborsClassifier()





comm.fit(x_train,y_train)

pro_comm_Pre = comm.predict_proba(x_train)


RightIndex=[]
for jj in range(100): #计算模型在不同分类阈值下的各项指标
    blo_comm_Pre = blo(pro_comm_Pre,jj)
    eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    RightIndex.append(abs(eva_comm['SEN'] - eva_comm['SPE']))
RightIndex=np.array(RightIndex,dtype=np.float16)

position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出
position=position.mean()


pro_comm_Pre = comm.predict_proba(x_test)
blo_comm_Pre = blo(pro_comm_Pre,position)  ##敏感性和特异性相差最小的点

eva_comm = evaluating_indicator(y_true=y_true_x, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
print('######################################')
print('时间窗',timesteps,gap,fore_len)

# print('RF')
print(eva_comm)  ##常规计算

# cumCI(y_true_x)



def turn_pro_comm_Pre(pro_comm_Pre,cpp,pos): # conformal_prediction_position -> cpp, position -> pos
     cpp=cpp/100;pos=pos/100
     pro_comm_Pre[pro_comm_Pre>=cpp] = ((pro_comm_Pre[pro_comm_Pre>=cpp]-cpp)/(1-cpp))*(1-pos)+pos
     pro_comm_Pre[pro_comm_Pre<cpp] = (pro_comm_Pre[pro_comm_Pre<cpp])/cpp*pos
     return pro_comm_Pre

comm.fit(x_train.append(x_test) ,y_train.append(y_true_x))
#joblib.dump(comm,'/home/guanjun/Liu/MODS_122_1116.pkl')

pro_comm_Pre=comm.predict_proba(x_train.append(x_test))
Corrected_probability_value=turn_pro_comm_Pre(pro_comm_Pre,19,50)
Corrected_probability_value=Corrected_probability_value[:,1]

comm_add_prd = lgb.LGBMClassifier()
comm_add_prd.fit(Corrected_probability_value.reshape(-1, 1) ,y_train.append(y_true_x))
joblib.dump(comm_add_prd,'/home/guanjun/Liu/MODS_122_1116_alldata.pkl')
################################zi dong tiao can####################################
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# def hyperopt_train_test(params):
# #    t = params['type']
# #    del params['type']
# #    if t == 'lgbm':
#     clf = lgb.LGBMClassifier(**params)
# #    elif t == 'svm':
# #        clf = SVC(**params)
# #    else:
# #        return 0
#     return cross_val_score(clf, x_train, y_train,cv=10, scoring = 'roc_auc').mean()
# #    return cross_val_score(clf, x_train, y_train,cv=5,scoring = 'balanecd_accurary').mean()

# #space = hp.choice('classifier_type', [
# space =       {
# #                'type': 'lgbm',
#                 'max_bin': hp.choice('max_bin', range(20,310)),
#                 'learning_rate': hp.uniform('learning_rate',0.0001,0.1),
#                 'n_estimators': hp.choice('n_estimators', range(10,500)),
#                 'max_depth': hp.choice('max_depth', range(10,200)),
#                 'num_leaves': hp.choice('num_leaves', range(50,300))
#                 }


# count = 0
# best = 0

# def f(params):
#     global best, count
#     count += 1
#     auc = hyperopt_train_test(params.copy())
#     if auc > best:
# #        print ('new best:', acc, 'using', params['type'])
#         print ('new best:', auc, 'using')
#         best = auc
#     if count % 5 == 0:
#         print ('iters:', count, ', auc:', auc, 'using', params)
#     return {'loss': -auc, 'status': STATUS_OK}

# trials = Trials()
# best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)
# print ('best:', best) 




# ##################################################################################
# ##################################################################################

# def RUN():   #根据训练集与验证集获取最优分类阈值
#     comm_cut_off=lgb.LGBMClassifier(**best)
#     comm_cut_off.fit(x_train , y_train) 
#     pro_comm_Pre = comm_cut_off.predict_proba(x_train)
#     RightIndex=[]
#     for jj in range(100): #计算模型在不同分类阈值下的各项指标
#         blo_comm_Pre = blo(pro_comm_Pre,jj)
#         eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
#         RightIndex.append(abs(eva_comm['SEN'] - eva_comm['SPE']))
#     RightIndex=np.array(RightIndex,dtype=np.float16)
#     position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出
#     position=position.mean()
    
#     blo_comm_Pre = blo(pro_comm_Pre,position)
#     eva_comm_train = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    
#     pro_comm_Pre = comm_cut_off.predict_proba(x_test)
#     blo_comm_Pre = blo(pro_comm_Pre,position)
#     eva_comm_test = evaluating_indicator(y_true=y_true_x, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
#     ###########################################
#     return  eva_comm_train,eva_comm_test
# ####################################################################################
# ####################################################################################


# eva_comm_train,eva_comm_test = RUN()

# print(eva_comm_train)
# print(eva_comm_test)
