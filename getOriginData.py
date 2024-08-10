import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, roc_curve,auc
import datetime
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from xgboost import XGBClassifier


# from scipy.stats import chi2

data_train = pd.read_csv('./train.csv')
data_test  = pd.read_csv('./test.csv')
data_train.head()


# 查找数据中的离散类型变量
def findDiscreateVariable(data):
    feas = data.columns
    print(feas)
    numerical_serial_fea = []    
    numerical_noserial_fea = []    
    for fea in feas:    
        temp = data[fea].nunique()    
        if temp <= 10 or data[fea].dtype=='object':    
            numerical_noserial_fea.append(fea)    
            continue    
        numerical_serial_fea.append(fea)    
    return numerical_serial_fea, numerical_noserial_fea    

# 离散变量的违约率分布图
def showDiscreatVariableDistrbution(attr='grade'):
    data_train.groupby(attr)['isDefault'].mean().plot(kind='bar', color='skyblue')
    plt.xlabel(attr)
    plt.ylabel('Default Rate')
    plt.title('Default Rate by ' + attr)
    plt.show()


# category_fea = ['term','grade','homeOwnership','verificationStatus','isDefault','initialListStatus','applicationType','n11','n12','subGrade','employmentLength']
# numerical_fea = ['loanAmnt','interestRate','installment','annualIncome','issueDate','purpose','postCode','regionCode','dti','delinquency_2years','ficoRangeLow','ficoRangeHigh','openAcc','pubRec','pubRecBankruptcies','revolBal','revolUtil','totalAcc','earliesCreditLine','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n13','n14']

#[numerical_fea, category_fea] = findDiscreateVariable(data_train)

# numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
# category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))


# 计算变量违约率的置信区间
def countConfidenceSection(data=data_train, attr='grade'):
    # 计算每个grade下的违约率
    default_rate = data.groupby(attr)['isDefault'].mean()
    
    # 计算每个grade下的样本数量
    sample_count = data.groupby(attr)['isDefault'].count()
    
    # 计算标准误差
    std_error = np.sqrt((default_rate * (1 - default_rate)) / sample_count)
    
    # 计算置信区间
    confidence_level = 0.95
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * std_error
    
    lower_bound = default_rate - margin_of_error
    upper_bound = default_rate + margin_of_error
    
    # 打印结果
    result = pd.DataFrame({
        'Default Rate': default_rate,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Sample Count': sample_count
    })
    print(result)
    
    # 可视化置信区间
    plt.figure(figsize=(10, 6))
    plt.errorbar(result.index, result['Default Rate'], yerr=margin_of_error, fmt='o', color='skyblue', ecolor='gray', elinewidth=2, capsize=4)
    plt.xlabel(attr)
    plt.ylabel('Default Rate')
    plt.title('Default Rate by '+ attr +' with 95% Confidence Interval')
    plt.grid(True)
    plt.show()

# 绘制连续变量的区间分布
def drawNumericalVarDistrbution():
    f = pd.melt(data_train, value_vars=numericalVars)    
    g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)    
    g = g.map(sns.histplot, "value", kde=True)

# 使用 KNN 填补缺失值
def KNNUpdateData(data_train=data_train):
    # 创建KNNImputer实例，n_neighbors是邻居的数量
    knn_imputer = KNNImputer(n_neighbors=2)
    # 对data_train进行插补
    data_train = pd.DataFrame(knn_imputer.fit_transform(data_train), columns=data_train.columns)

# 绘制缺失值
def showNan():
    missing = data_train.isnull().sum()/len(data_train)    
    missing = missing[missing > 0]    
    missing.sort_values(inplace=True)    
    missing.plot.bar()

# 绘制同值率
def calculate_homogeneity(dataframe=data_train):
    homogeneity_rates = {}
    for column in dataframe.columns:
        # 计算每个唯一值的频率
        max_frequency = dataframe[column].value_counts(normalize=True).max()
        homogeneity_rates[column] = max_frequency
    return homogeneity_rates

# 删除同值率为 100%
def removeHomogeneity(data_train=data_train, data_test=data_test):
    # 计算同值率   
    column_homogeneity_rates = calculate_homogeneity(data_train)
    # 打印结果
    for column, rate in column_homogeneity_rates.items():
        if rate == 1:
            print(f"列 '{column}' 的同值率: {rate:.2%}")
            data_train.drop(columns=column, inplace=True)
            data_test.drop(columns=column, inplace=True)
    return data_train, data_test

# data_train = removeHomogeneity(data_train)

# 同值率
def homogeneityAnalyse():    
    # 计算同值率   
    column_homogeneity_rates = calculate_homogeneity(data_train)
    # 打印结果
    for column, rate in column_homogeneity_rates.items():    
        if rate > 0.9:
            # 计算每个类别的违约率
            category_default_rates = data_train[[column,'isDefault']].groupby(column)['isDefault'].apply(lambda x: x.sum()/x.count())
            # 绘制条形图
            plt.figure(figsize=(10, 6))
            sns.barplot(x=category_default_rates.index, y=category_default_rates.values)
            plt.title('Default Rate Distribution for '+column)
            plt.xlabel(column)
            plt.ylabel('Default Rate (%)')

            # 显示图形
            plt.show()

#def corrAnalysis():

def employmentLength_to_int(s):    
    if pd.isnull(s):    
        return s    
    else:    
        return np.int8(s.split()[0])    
    
# 标准差异常值
def find_outliers_by_3segama(data_train=data_train):
    VarianceThreshold(threshold=3).fit_transform(data_train)

# 分箱
def KBinsDiscretizerhandle(data_train=data_train, data_test=data_test, feat='loanAmnt', strategy='quantile'):
    X = pd.DataFrame(data_train[feat])
    y = data_train['isDefault']

    kbd = KBinsDiscretizer(n_bins=10,encode='ordinal', strategy=strategy)
    X_binned = kbd.fit_transform(X,y)
    X_test_binned = kbd.fit_transform(pd.DataFrame(data_test[feat]))
    # 将分箱结果添加回 DataFrame
    data_train[feat+'_binned'] = X_binned.astype(int)
    data_test[feat+'_binned'] = X_test_binned.astype(int)

    # 显示分箱结果
    print("\n对"+feat+"的分箱结果：")
    print(data_train[[feat, feat+'_binned']].head())

    # 统计每个箱的样本数
    bin_counts = data_train[feat+'_binned'].value_counts().sort_index()

    # 显示每个箱的样本数
    print("\n每个箱的样本数：")
    print(bin_counts)
    data_train.drop(columns=feat, inplace=True)
    data_test.drop(columns=feat, inplace=True)
    return data_train, data_test

# 卡方分箱
# def chiDiscretizerdef(data_train,feat, max_bins=5):
#     """
#     data: 待分箱的连续变量
#     target: 目标变量
#     max_bins: 最大分箱数
#     """
#     def chi2_stat(arr):
#         """计算卡方统计量"""
#         if len(arr.shape) == 1:
#             arr = np.array(arr, dtype=np.float64).reshape(-1, 1)
#         n_total = arr.sum()
#         n_col = arr.sum(axis=0)
#         n_row = arr.sum(axis=1)
#         E = np.outer(n_row, n_col) / n_total
#         chi2 = ((arr - E)**2 / E).sum()
#         return chi2

#     # 初始化分箱
#     data = data_train[feat]
#     target = data_train['isDefault']
#     n_samples = len(data)
#     bins = np.unique(data)
#     while len(bins) > max_bins:
#         chi2_vals = []
#         for i in range(len(bins) - 1):
#             left_bin = data[(data >= bins[i]) & (data < bins[i + 1])]
#             right_bin = data[data >= bins[i + 1]]
#             left_target = target[(data >= bins[i]) & (data < bins[i + 1])]
#             right_target = target[data >= bins[i + 1]]
#             chi2_vals.append(chi2_stat(np.array([left_target, right_target])))
#         min_chi2_index = np.argmin(chi2_vals)
#         bins = np.delete(bins, min_chi2_index + 1)
#     data_train[feat+'_binned'] = pd.cut(data_train[feat], bins, include_lowest=True)
#     data_train.drop(columns=feat, inplace=True)
#     return data

# 特征数据处理 高纬度类别
def labelEncodeData(data_train=data_train, data_test=data_test):
    for col in ['employmentTitle', 'postCode', 'title']:
        le = LabelEncoder()
        le.fit(list(data_train[col].astype(str).values) + list(data_test[col].astype(str).values))
        data_train[col] = le.transform(list(data_train[col].astype(str).values))
        data_test[col] = le.transform(list(data_test[col].astype(str).values))
    return data_train, data_test

# 数据清洗
def handleData(data_train=data_train, data_test=data_test):
    # category_fea.remove('isDefault')
    # category_fea.remove('subGrade')
    # test_cat=category_fea.copy()
    # test_cat.remove('isDefault')
    # train_numerical_fea = numerical_fea.copy()
    
    # data_train.set_index(keys='id')
    # data_test.set_index(keys='id')
    data_train.drop('id', axis=1, inplace=True)
    # data_test.drop('id',axis=1, inplace=True)
    
    # 缺失值
    # data_train = data_train.fillna(0)
    # data_test = data_test.fillna(0)
    # data_train = data_train.fillna(axis=0,method='ffill')
    # data_test = data_test.fillna(axis=0,method='ffill')
    # data_train = data_train.fillna(axis=0,method='bfill',limit=2)
    # data_test = data_test.fillna(axis=0,method='bfill',limit=2)

    # 负值处理
    for data in [data_train]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        mask = (data[numeric_cols] < 0).any(axis=1)
        data = data[~mask & ~data[numeric_cols].isna().any(axis=1)]
    
    # 同值率处理
    [data_train, data_test] = removeHomogeneity(data_train, data_test)
    
    # grade 处理
    for data in [data_train, data_test]:
        data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})

    # 日期
    for data in [data_train, data_test]:    
        data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')    
        startdate = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d')    
        #构造时间特征    
        data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days
        data.drop(columns='issueDate', inplace=True)
    
    # earliesCreditLine
    for data in [data_train, data_test]:    
        data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

    # 工作年限
    for data in [data_train, data_test]:    
        data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)    
        data['employmentLength'].replace(to_replace='< 1 year', value='0 years', inplace=True)    
        data['employmentLength'].replace(0, '0 years', inplace=True)    
        data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

    [train_numerical_fea, category_fea] = findDiscreateVariable(data_train)
    test_cat = category_fea.copy()
    test_cat.remove('isDefault')
    test_cat.remove('subGrade')
    
    
    #按照众数填充类别型特征  
    data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())    
    data_test[test_cat] = data_test[test_cat].fillna(data_test[test_cat].mode())    
    
    #按照平均数填充数值型特征   
    data_train[train_numerical_fea] = data_train[train_numerical_fea].fillna(data_train[train_numerical_fea].median())    
    data_test[train_numerical_fea] = data_test[train_numerical_fea].fillna(data_train[train_numerical_fea].median())    
    
    data_train.dropna()
    
    # 登记处理 使用更细分的subGrade 删除 grade
    for data in [data_train, data_test]:    
        # data.drop(columns='grade', inplace=True)
        data['subGrade'] = data['subGrade'].map({
            'A1':1,
            'A2':2,
            'A3':3,
            'A4':4,
            'A5':5,
            'B1':6,
            'B2':7,
            'B3':8,
            'B4':9,
            'B5':10,
            'C1':11,
            'C2':12,
            'C3':13,
            'C4':14,
            'C5':15,
            'D1':16,
            'D2':17,
            'D3':18,
            'D4':19,
            'D5':20,
            'E1':21,
            'E2':22,
            'E3':23,
            'E4':24,
            'E5':25,
            'F1':26,
            'F2':27,
            'F3':28,
            'F4':30,
            'F5':32,
            'G1':29,
            'G2':31,
            'G3':33,
            'G4':34,
            'G5':35})
    
    # 类型数在2之上，又不是高维稀疏的,且纯分类特征    
    data_train = pd.get_dummies(data_train, columns=['homeOwnership', 'verificationStatus', 'purpose', 'regionCode'])
    data_test = pd.get_dummies(data_test, columns=['homeOwnership', 'verificationStatus', 'purpose', 'regionCode'])

    
    [data_train, data_test] = KBinsDiscretizerhandle(data_train, data_test, 'loanAmnt')
    # [data_train, data_test] = KBinsDiscretizerhandle(data_train, data_test, 'annualIncome')
    
    [data_train, data_test] = labelEncodeData(data_train, data_test)

    return data_train, data_test

# 清洗后的数据
[data_train_c, data_test_c] = handleData(data_train, data_test)

# 衍生特征
def generateFeature(data_train,data_test):
    # 申报到贷款首次发放的时间差
    for data in [data_train, data_test]:
        data['creditAge']= data['issueDateDT'] - data['earliesCreditLine']
        data['openAccUtilization'] = data['openAcc'] / data['totalAcc']
        data.drop(columns=['issueDateDT','earliesCreditLine', 'openAcc', 'totalAcc'], inplace=True)
    return [data_train, data_test]
[data_train_g, data_test_g] = generateFeature(data_train_c,data_test_c)

# 删除缺失值
data_train_g = data_train_g.dropna()

# 数据标准化
def standrdData(data_train,data_test):
    data_test.drop(columns='id', inplace=True)
    # 选择需要标准化的列，这里我们选择数值型的列
    numerical_cols = data_test.select_dtypes(include=['float64', 'int64']).columns

    # 初始化StandardScaler
    scaler = StandardScaler()

    # 对选择的列进行Z-score标准化
    data_train[numerical_cols] = scaler.fit_transform(data_train[numerical_cols])
    data_test[numerical_cols] = scaler.fit_transform(data_test[numerical_cols])

    return data_train,data_test

[data_train_s, data_test_s] = standrdData(data_train_g, data_test_g)

# 计算特征相似度
def calculate_vif(data_train=data_train):
    corr_matrix = data_train.corr()
    # 全为 nan 的列 无法使用 fillna 处理
    corr_matrix.fillna(corr_matrix.mean(), inplace=True)
    corr_matrix = corr_matrix.dropna(axis=1, how='all')  # 删除完全为 NaN 的列
    corr_matrix = corr_matrix.dropna(axis=0, how='all')  # 删除完全为 NaN 的行
    
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    # plt.title('Correlation Matrix Heatmap')
    # plt.show()

    numeric_cols = corr_matrix.select_dtypes(include=['float64', 'int64']).columns

    
    # 计算VIF值
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_cols
    vif_data['VIF'] = [variance_inflation_factor(corr_matrix.values, i) for i in range(corr_matrix.shape[1])]
    
    return vif_data

#!SECTION 多重共线性特征筛选
def remove_high_vif_features(df, threshold=5):
    while True:
        vif_data = calculate_vif(df)
        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            print(f'Removing feature: {max_vif_feature} with VIF: {max_vif}')
            df = df.drop(columns=[max_vif_feature])
        else:
            break
    return df
# 过滤掉 VIF > 5 的特征
filter_feature_vif = remove_high_vif_features(data_train_s).columns
print(filter_feature_vif)
data_train_s_vif = data_train_s[filter_feature_vif]
data_test_s_vif = data_test_s[filter_feature_vif.drop('isDefault')]

# Removing feature: n2 with VIF: inf
# Removing feature: ficoRangeLow with VIF: 290554814669064.25
# Removing feature: loanAmnt with VIF: 20582.159128155923
# Removing feature: n9 with VIF: 17497.374397975196
# Removing feature: n10 with VIF: 6743.547544702596
# Removing feature: n8 with VIF: 2571.6580091866554
# Removing feature: interestRate with VIF: 682.9543721740638
# Removing feature: n7 with VIF: 468.0459757239794
# Removing feature: n1 with VIF: 281.3247203724065
# Removing feature: totalAcc with VIF: 185.25213169039966
# Removing feature: n4 with VIF: 46.18315953883162
# Removing feature: openAcc with VIF: 20.68838250649326
# Removing feature: ficoRangeHigh with VIF: 10.03015088551691
# ['id', 'term', 'interestRate', 'installment', 'employmentTitle','employmentLength', 'annualIncome', 'isDefault', 'postCode', 'dti','delinquency_2years', 'ficoRangeHigh', 'pubRec', 'pubRecBankruptcies','revolBal', 'revolUtil', 'initialListStatus', 'applicationType', 'n0','n5', 'n6', 'n9', 'n11', 'n12', 'n13', 'n14', 'homeOwnership_1','homeOwnership_2', 'homeOwnership_4', 'homeOwnership_5','verificationStatus_1', 'verificationStatus_2', 'purpose_1','purpose_2', 'purpose_3', 'purpose_4', 'purpose_5', 'purpose_6','purpose_7', 'purpose_8', 'purpose_9', 'purpose_10', 'purpose_11','purpose_12', 'regionCode_0', 'regionCode_1', 'regionCode_2','regionCode_3', 'regionCode_4', 'regionCode_5', 'regionCode_6','regionCode_7', 'regionCode_9', 'regionCode_10', 'regionCode_11','regionCode_12', 'regionCode_13', 'regionCode_14', 'regionCode_15','regionCode_16', 'regionCode_17', 'regionCode_18', 'regionCode_19','regionCode_20', 'regionCode_21', 'regionCode_22', 'regionCode_23','regionCode_24', 'regionCode_25', 'regionCode_26', 'regionCode_27','regionCode_28', 'regionCode_29', 'regionCode_30', 'regionCode_31','regionCode_32', 'regionCode_33', 'regionCode_34', 'regionCode_35','regionCode_36', 'regionCode_37', 'regionCode_38', 'regionCode_39','regionCode_40', 'regionCode_41', 'regionCode_42', 'regionCode_43','regionCode_44', 'regionCode_45', 'regionCode_46', 'regionCode_47','regionCode_48', 'regionCode_49', 'creditAge', 'openAccUtilization']

#!SECTION 方差筛选特征
def selectFeatureByVariance(data_train=data_train):
    # 定义方差筛选器，阈值设为 0.5
    selector = VarianceThreshold(threshold=0.5)
    new_data = data_train.select_dtypes(include=['float64', 'int64'])
    # 应用方差筛选
    numeric_features = selector.fit_transform(new_data)
    selected_features = new_data.columns[selector.get_support()]
    print(selected_features)
    return selected_features
#['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade','employmentTitle', 'homeOwnership', 'annualIncome','verificationStatus', 'purpose', 'postCode', 'regionCode', 'dti','delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc','pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc','initialListStatus', 'applicationType', 'earliesCreditLine', 'title','n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10','n11', 'n12', 'n13', 'n14', 'issueDateDT']


# 基于相关性属性选择
def coorFeature(data_train=data_train, threshold=0.05):
    corr_matrix = data_train.corr()
    # 全为 nan 的列 无法使用 fillna 处理
    corr_matrix.fillna(corr_matrix.mean(), inplace=True)
    corr_matrix = corr_matrix.dropna(axis=1, how='all')  # 删除完全为 NaN 的列
    corr_matrix = corr_matrix.dropna(axis=0, how='all')  # 删除完全为 NaN 的行
    corr_with_target = corr_matrix['isDefault'].drop('isDefault')
    print(corr_with_target)
    selected_features = corr_with_target[abs(corr_with_target) > threshold].index
    print(selected_features)
    return selected_features
# coorFeature(data_train_s.select_dtypes(include=['float64', 'int64']))
#['loanAmnt', 'term', 'interestRate', 'installment', 'grade','homeOwnership', 'verificationStatus', 'dti', 'ficoRangeLow','ficoRangeHigh', 'revolUtil', 'policyCode', 'n2', 'n3', 'n9', 'n14','issueDateDT']
coorFeature(data_train_s)
#['loanAmnt', 'term', 'interestRate', 'installment', 'subGrade', 'dti','ficoRangeLow', 'ficoRangeHigh', 'revolUtil', 'n2', 'n3', 'n9', 'n14','issueDateDT', 'homeOwnership_0', 'homeOwnership_1','verificationStatus_0', 'verificationStatus_2']

#Pearson 相关系数
def pearsonFeature(train=data_train_s, k=15):
    X = train.drop(columns=['isDefault'])
    y = train['isDefault']
    selector =  SelectKBest(k=k, score_func=f_classif)
    selector.fit_transform(X,y)
    selected_features = train.columns[selector.get_support(indices=True)]
    return selected_features
# print(pearsonFeature(data_train_s.select_dtypes(exclude=['object'])))
#['loanAmnt', 'term', 'interestRate', 'grade', 'homeOwnership','verificationStatus', 'regionCode', 'delinquency_2years','ficoRangeLow', 'revolBal', 'n1', 'n2', 'n8', 'n13', 'n14']
pearsonFeature(data_train_s)
#['loanAmnt', 'term', 'interestRate', 'subGrade', 'postCode','delinquency_2years', 'ficoRangeLow', 'n1', 'n2', 'n8', 'n13','issueDateDT', 'homeOwnership_0', 'homeOwnership_5','verificationStatus_1']

#信息增益法筛选特征(互信息)
def mutualFeature(data_train=data_train, threshold = 0.02):
    X = data_train.drop(columns=['isDefault'])
    y = data_train['isDefault']
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame(mi, index=X.columns, columns=['Mutual Information'])
    mi_df.sort_values(by='Mutual Information', ascending=False, inplace=True)
    selected_features = mi_df[mi_df['Mutual Information'] > threshold].index
    print(mi_df)
    print(selected_features)
    return selected_features
#mutualFeature(data_train_s.select_dtypes(include=['float64', 'int64']))

#                     Mutual Information
# term                          0.078822
# initialListStatus             0.062089
# grade                         0.054291
# homeOwnership                 0.042792
# interestRate                  0.039923
# verificationStatus            0.039389
# pubRecBankruptcies            0.039044
# purpose                       0.036107
# pubRec                        0.034717
# delinquency_2years            0.031518
# installment                   0.029572
# n0                            0.028226
# applicationType               0.022367
# n14                           0.021749
#---
# n13                           0.019903
# ficoRangeLow                  0.014857
# ficoRangeHigh                 0.014208
# n1                            0.011056
# employmentTitle               0.009311
# title                         0.009023
# n9                            0.008808
# loanAmnt                      0.008728
# n3                            0.008687
# n2                            0.008595
# n4                            0.007027
# n7                            0.005996
# regionCode                    0.005546
# dti                           0.005403
# n5                            0.005301
# n12                           0.005197
# n6                            0.005189

mutualFeature(data_train_s)
# for key in disCreateVars:
#     countConfidenceSection(attr=key)

# 树模型的特征选择 (随机森林)
def gradientBoostingFeature(data_train=data_train, n_features_to_select=15):
    X = data_train.drop(columns=['isDefault'])
    y = data_train['isDefault']
    # 选择基模型
    model = RandomForestClassifier()

    # 设置RFE参数
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    # 拟合RFE
    rfe.fit(X, y)

    # 查看所选特征
    selected_features = X.columns[rfe.support_]
    print("Selected features for binning:", selected_features)
gradientBoostingFeature(data_train_s)

# 卡方特征筛选
def chi2FilterFeature(data_train_s):
    # 检查特征是否都是非负的
    if not data_train_s.drop(columns='isDefault').applymap(lambda x: x >= 0).all().all():
        # 如果有负数，将特征转换为绝对值
        data_train_s.loc[:, data_train_s.columns != 'isDefault'] = data_train_s.drop(columns='isDefault').abs()

    # 使用SelectKBest和chi2进行特征选择
    selector = SelectKBest(chi2, k=15)
    selector.fit_transform(data_train_s.drop(columns='isDefault'), data_train_s['isDefault'])

    # 获取选择的特征的索引
    selected_features_indices = selector.get_support(indices=True)

    # 打印选择的特征名称
    print(data_train_s.columns[selected_features_indices])
chi2FilterFeature(data_train_s)
#['term', 'interestRate', 'subGrade', 'employmentLength','delinquency_2years', 'ficoRangeLow', 'pubRec', 'pubRecBankruptcies','initialListStatus', 'issueDateDT', 'homeOwnership_0','homeOwnership_5', 'verificationStatus_1', 'purpose_0', 'purpose_3']

# 异常值处理
def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data
data_train_t = data_train.copy()
for fea in ['loanAmnt','interestRate','installment','annualIncome','regionCode','dti','delinquency_2years','ficoRangeLow','ficoRangeHigh','openAcc','pubRec','pubRecBankruptcies','revolBal','revolUtil','totalAcc','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n13','n14']:
    data_train_t = find_outliers_by_3segama(data_train_t,fea)
    print(data_train_t[fea+'_outliers'].value_counts())
    print(data_train_t.groupby(fea+'_outliers')['isDefault'].sum())
    errordf = data_train_t.groupby(fea+'_outliers')['isDefault'].sum()/data_train_t.groupby(fea+'_outliers')['isDefault'].count()
    print(errordf)
    print('*'*10)
    if errordf.get('xx') is not None and (abs(errordf['异常值'] - errordf['正常值']) < 0.03):
        data_train = data_train[data_train[fea+'_outliers']=='正常值']
        data_train = data_train.reset_index(drop=True) 

data_train_t['count_异常值'] = data_train_t.apply(lambda row: row.apply(lambda x: x == '异常值').sum())
indexes_to_drop = data_train_t[data_train_t['count_异常值'] > 2].index
data_train_t_cleaned = data_train_t.drop(indexes_to_drop)

# 组合特征筛选
def FeatureFilter(data_train, data_test, num=20):
    feat_pool = np.array([])
    # 方差筛选
    feat_pool = np.concatenate((feat_pool, selectFeatureByVariance(data_train)), axis=0)
    # 相关系数法
    feat_pool = np.concatenate((feat_pool,coorFeature(data_train)), axis=0)
    feat_pool = np.concatenate((feat_pool,pearsonFeature(data_train, num)), axis=0)
    # 信息增益法
    feat_pool = np.concatenate((feat_pool,mutualFeature(data_train, num)), axis=0)
    counter = Counter(feat_pool)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print('sorted_items:'+ sorted_items)
    return data_train[counter.items()], data_test[counter.items()]

# 精确度、准确率、召回率、F1、AUC 评价
def judgeModel(y_train, y_pred, X_train, model):
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='binary')  # 对于二分类问题使用 'binary'
    recall = recall_score(y_train, y_pred, average='binary')
    f1 = f1_score(y_train, y_pred, average='binary')

    # 计算AUC需要概率估计
    y_pred_proba = model.predict_proba(X_train)[:, 1]  # 获取正类的预测概率
    auc = roc_auc_score(y_train, y_pred_proba)

    # 打印性能指标
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")






def trainData(X,y, Classifier = RandomForestClassifier, param_grid={}, gridAdvance = False, params = {}):
    # X = X.head(10000)
    # y = y.head(10000)
    # 初始化模型
    model = Classifier(random_state=42, n_estimators=100)

    # 使用SMOTE平衡数据
    smote = BorderlineSMOTE(random_state=42)

    # 定义Pipeline
    pipeline = imbpipeline(steps=[('smote', smote), ('model', model)])
    
    # 执行交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred = []
    y_pred_proba = []
    # 自定义评分函数，保存每个折的模型
    def custom_scoring(estimator, X_test, y_test):
        y_pred.append(estimator.predict(X_test))
        # 计算AUC需要概率估计
        y_pred_proba_c = estimator.predict_proba(X_test)[:, 1]
        y_pred_proba.append(y_pred_proba_c)  # 获取正类的预测概率
        # 计算ROC曲线的FPR和TPR
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_c)
        # KS值是TPR和FPR曲线之间的最大垂直距离
        ks = max(x2 - x1 for x1, x2 in zip(fpr, tpr))
        return ks
    
    if gridAdvance:
        # 创建GridSearchCV实例
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, scoring=custom_scoring)

        # 执行网格搜索
        grid_search.fit(X, y)

        # 获取最优参数
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        y_pred = grid_search.predict(X)
        y_pred_proba = grid_search.predict_proba(X)[:, 1]
    else:
        cross_val_score(pipeline, X, y, cv=kf, scoring=custom_scoring, params=params)
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred_proba = np.concatenate(y_pred_proba, axis=0)

    return [y, y_pred, y_pred_proba]

model_feat = ['creditAge','term','installment','annualIncome','postCode','delinquency_2years','ficoRangeHigh','pubRec','n12','n14','homeOwnership_5','verificationStatus_1','verificationStatus_2','purpose_3','regionCode_50','dti','interestRate','grade','subGrade','ficoRangeLow','revolUtil','n2','n3','n9','loanAmnt_binned','openAccUtilization','initialListStatus','n0']
# trainData(data=data_train_s[model_feat], Classifier= RandomForestClassifier, model_name='rf')
# trainData(data=data_train_s[model_feat], Classifier= GradientBoostingClassifier, model_name='gb')
# trainData(data=data_train_s[model_feat], Classifier= lgb.LGBMClassifier, model_name='lgb')

# 精确度、准确率、召回率、F1、AUC 评价
def judgeModel(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # 对于二分类问题使用 'binary'
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    # 计算AUC需要概率估计
    auc_score = roc_auc_score(y_test, y_pred_proba)
    # 计算KS值
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc =  auc(fpr, tpr)
    ks = max(x2 - x1 for x1, x2 in zip(fpr, tpr))
    # 打印性能指标
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"KS Value: {ks}")

    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return [
        accuracy,
        precision,
        recall,
        f1,
        auc_score,
        ks
    ]

models_result_rf = []
models_result_lgb = []
models_result_gb = []

models_result_rf = trainData(X=data_train_s[model_feat], y=data_train_s['isDefault'], Classifier= RandomForestClassifier,
    param_grid = {
    'model__n_estimators': [100, 200],  # 随机森林的树的数量
    'model__max_depth': [None, 10, 20],  # 树的最大深度
    # 添加其他需要调优的参数
    })
judgeModel(*models_result_rf)


models_result_lgb = trainData(X=data_train_s[model_feat], y=data_train_s['isDefault'], Classifier= LGBMClassifier)
judgeModel(*models_result_lgb)


models_result_gb= trainData(X=data_train_s[model_feat], y=data_train_s['isDefault'], Classifier= GradientBoostingClassifier)
judgeModel(*models_result_gb)

models_result_xgb= trainData(X=data_train_s[model_feat], y=data_test_s['isDefault'], Classifier=XGBClassifier, 
    param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__min_child_weight': [1, 5, 10],
    'model__gamma': [0, 0.5, 1],
    'model__subsample': [0.8, 1],
    'model__colsample_bytree': [0.8, 1],
    },
    params={
        'objective': 'binary:logistic',
        'booster': 'gbtree'
    })
judgeModel(*models_result_xgb)

def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(i, train_index, valid_index)
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_x , ntree_limit=model.best_ntree_limit)
                 
        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        if clf_name == 'rf':
            params = {'n_estimators':100, 'random_state': 42}
            model = clf(**params)
            model.fit(trn_x, trn_y)
            # 模型评估
            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)

        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
        
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(LGBMClassifier, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test


def rf_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(RandomForestClassifier, x_train, y_train, x_test, "rf")
    return xgb_train, xgb_test


[rf_train,rf_test] = rf_model(data_train_s[model_feat], data_train_s['isDefault'], data_test_s[model_feat])
[lgb_train, lgb_test] = lgb_model(data_train_s[model_feat], data_train_s['isDefault'], data_test_s[model_feat])


