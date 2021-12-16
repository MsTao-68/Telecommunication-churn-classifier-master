# !usr\bin\python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/9/15

#载入包；设置;载入数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid',font_scale=1.3)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False
import warnings
from scipy import stats
#标准化
from sklearn.preprocessing import StandardScaler
import sklearn #特征工程
from sklearn import preprocessing  #数据预处理
from sklearn.preprocessing import LabelEncoder #编码转换
from sklearn.model_selection import StratifiedShuffleSplit #分层抽样
from sklearn.model_selection import train_test_split #数据集训练集划分
# 训练分类模型
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.svm import SVC#支持向量机
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.neighbors import KNeighborsClassifier #k邻近算法
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import AdaBoostClassifier #分类器算法
from sklearn.ensemble import GradientBoostingClassifier #梯度提升
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier # 岭
from sklearn.neural_network import MLPClassifier #神经网络
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import time
#sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = StandardScaler(copy=False)
Classifiers = [["Random Forest",RandomForestClassifier()],
             ["Support Vector Machine",SVC()],
             ["LogisticRegression",LogisticRegression()],
             ["KNeighbor",KNeighborsClassifier(n_neighbors=5)],
             ["Naive Bayes",GaussianNB()],
             ["Decision Tree",DecisionTreeClassifier()],
             ["GradientBoostingClassifier",GradientBoostingClassifier()],
             ["XGB",XGBClassifier()],
             ["CatBoost",CatBoostClassifier(logging_level='Silent')],
              ['RidgeClassifier',RidgeClassifier()],
               ['MLPClassifier',MLPClassifier(solver='lbfgs',activation = 'tanh',
                    max_iter = 50,alpha = 0.001,
                    hidden_layer_sizes = (10,30),
                    random_state = 1,verbose = True)],
               ['SGDClassifier',SGDClassifier()],
               ['XGBClassifier',XGBClassifier()],
               ['BaggingClassifier',BaggingClassifier()],
               ['XGBClassifier',XGBClassifier()]
              ]
warnings.filterwarnings('ignore')

# 将分类数据转化成整数编码
# 获取分类变量的标签值
def Labs(x):
    print(x,"--",df1[x].unique())
def labelencoder(x):
    df1[x]=LabelEncoder().fit_transform(df1[x])


df1 = pd.read_csv("E:\\dasein_py\\Telecommunication_da\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df1.info())   # 查看数据集详情描述
print(df1.shape)    # 查看数据集数据量
quantative = [i for i in df1.columns if df1[i].dtype!=object]
quanlitive = [i for i in df1.columns if df1[i].dtype==object]
print("Quantative counts:{}, Quanlitive counts:{}".format(len(quantative),len(quanlitive))) # 查看定性和定量特征
print(df1.describe())       # 定量数据描述性统计
df1.drop('customerID',axis=1,inplace=True) #drop colName: CustomerID
print(df1.head(3))  # 查看数据
total = df1.isnull().sum()
null_percentage = total/df1.isnull().count()
print('数据缺失率：',null_percentage)
print(df1.duplicated().sum())
df1=df1.drop_duplicates(subset=None, keep='first',inplace=False)    # 去除重复性数据
df1['TotalCharges'] = df1['TotalCharges'].apply(pd.to_numeric, errors='coerce') # TotalCharges应该是数值型，需要强制类型转换
print('TotalCharges数据分布')
plt.figure(figsize=(14,5))
plt.plot(color='#00338D')
#1
plt.subplot(1,3,1)
plt.title("TotalCharges distplot")
sns.distplot(df1.TotalCharges)
#2
plt.subplot(1,3,2)
plt.title("Churn Distplot")
sns.distplot(df1[df1.Churn=='No']['TotalCharges'])
#2
plt.subplot(1,3,3)
plt.title("Churn + TotalCharges Distplot")
sns.distplot(df1[df1['Churn']=='Yes']['TotalCharges'])
plt.show()
df1.fillna({'TotalCharges':df1.TotalCharges.median()},inplace=True) # 中值填充
#重编码‘Churn’，定性转定量的哑变量
#df1.Churn=df1.Churn.map({'Yes':1,'No':0})
df1.Churn.replace(to_replace='Yes',value=1,inplace=True)
df1.Churn.replace(to_replace='No',value=0,inplace=True)
# df1.Churn.isnull().sum()
# print(df1.Churn.describe())

Churn_Count=df1.Churn.value_counts()
Churn_Lab=df1.Churn.value_counts().index
plt.figure(figsize=(5,5))
plt.pie(Churn_Count,labels=Churn_Lab,
        colors=["#00338D","red"],
        explode=(0.3,0),
        autopct="%1.1f%%",
        shadow=True)
plt.title("Customer Churn Pie Chart")
plt.show()

plt.figure(figsize=(4,6))
plt.plot(color='#00338D')
fig = sns.boxplot(x="Churn",y="tenure",data=df1)
plt.title("tenure - Churn Boxplot")
fig.axis(ymin=0,ymax=80)
plt.show()
df2 = df1.apply(lambda x:pd.factorize(x)[0]) #转换成因子
# df2.head(5)
var = list(df2.columns)
var.remove("Churn")
var.remove("tenure")
var.remove("MonthlyCharges")
var.remove("TotalCharges")
plt.figure(figsize=(20,25))
a=0
for item in var:
    a+=1
    plt.subplot(4,5,a)
    plt.title('Barplot by '+ item)
    sns.countplot(x=item,data=df2,
                 color="#00338D")
#sns.countplot(x=None, y=None,
#hue=None, data=None, order=None,
#hue_order=None, orient=None, color=None,
#palette=None, saturation=0.75, dodge=True, ax=None, **kwargs)
df2.drop("gender",axis=1,inplace=True)
corr = df2.corr()
plt.figure(figsize=(15,15))
sns.set(font_scale=1.25)
ax=sns.heatmap(corr,
               xticklabels=corr.columns,
               yticklabels=corr.columns,
               linewidths=0.6,annot=True,
               cbar=True,cmap="rainbow",fmt='.4f',
               annot_kws={'size': 10})
plt.title("Corr Heatmap")
plt.savefig("Corr Heatmap.png",dpi=400)
plt.show()
#独热编码
df_onehot = pd.get_dummies(df1.iloc[:,:])
df2.drop("PhoneService",axis=1,inplace=True)
var.remove('PhoneService')
var.remove('gender')
#交叉分析
print('kf_var与Churn的进行交叉分析','\n')
for item in var:
    print("---------Churn by {}---------".format(item))
    print(pd.crosstab(df2.Churn,df2[item],normalize=0),'\n')
print(df1[["MonthlyCharges","TotalCharges"]])   # 量纲有问题
scaler.fit_transform(df1[['MonthlyCharges','TotalCharges']]) #拟合数据
df1[['MonthlyCharges','TotalCharges']]=scaler.transform(df1[['MonthlyCharges','TotalCharges']]) #数据标准化
# df1[['MonthlyCharges','TotalCharges']].head()

df_obj = df1.select_dtypes(['object'])
print(list(map(Labs,df_obj)))
df1.replace(to_replace='No internet service',value = 'No',inplace=True)
df1.replace(to_replace='No phone service',value='No',inplace=True)
df_obj = df1.select_dtypes(['object'])
print(list(map(Labs,df_obj)))

for i in range (len(df_obj.columns)):
    labelencoder(df_obj.columns[i])
print(list(map(Labs,df_obj.columns)))

#处理样本不平衡，分拆变量
df1.drop("gender",axis=1,inplace=True)
df1.drop("PhoneService",axis=1,inplace=True)

sss=StratifiedShuffleSplit(n_splits=5,test_size=.2,random_state=0)
print(sss)
x=df1[var]
y=df1['Churn'].values
print(sss.split(x,y))
print("训练数据和测试数据被分成的份数：",sss.get_n_splits(x,y))
#拆分训练集和测试集
for train_index,test_index,in sss.split(x,y):
    print("train:",train_index,"test:",test_index)
    x_train_,x_test_=x.iloc[train_index],x.iloc[test_index]
    y_train_,y_test_ = y[train_index],y[test_index]
print("分层抽样数据特征：",x.shape,"train特征:",x_train_.shape,"test特征：",x_test_.shape)
print("分层抽样数据特征：",y.shape,"train特征:",y_train_.shape,"test特征：",y_test_.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=22)


Classify_result=[]
names=[]
prediction=[]
for name,classifier in Classifiers:
    classifier=classifier
    t1 = time.time()
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    t2=time.time()
    precision=precision_score(y_test,y_pred)
    f1score = f1_score(y_test, y_pred)
    time_diff = t2 -t1
    class_eva=pd.DataFrame([precision,f1score,time_diff])
    Classify_result.append(class_eva)
    name=pd.Series(name)
    names.append(name)
    y_pred=pd.Series(y_pred)
    prediction.append(y_pred)
names = pd.DataFrame(names)
names=names[0].tolist()
result = pd.concat(Classify_result,axis=1)
result.columns =names
result.index = ["precision",'f1score',"time_diff"]
print(result.T)
