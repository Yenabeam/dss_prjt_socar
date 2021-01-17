# 쏘카 보험 사기 탐지 머신러닝 프로젝트
![image](https://user-images.githubusercontent.com/72847093/104838734-3476b900-5900-11eb-9428-96d19d7840d8.png)

## 개요 

### 1.1  프로젝트 주제
- 보험금을 목적으로 한 사기 집단의 렌터카를 이용한 사기 수법이 기승함. 
- 이에 쏘카의 사고 데이터를 통해 Fraud 유저에 대한 예측을 실현하고자 함 

### 1.2 프로젝트 진행순서
1. DATA SET 
2. EDA 
3. 데이터 전처리 및 가공 
4. 모델 학습/성능 평가 
5. 결론 
 
### 1.3 시작에 앞서
- 본 프로젝트를 진행하기 위해서는 __Python 3__ 이상의 버젼과 다음의 설치가 필요합니다.
```
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install sklearn
pip install lightgbm 
pip install imblearn
pip install warnings
pip install sweetviz
pip install statsmodels
```

## Setting 

### 1.1 환경설정
```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from statsmodels.stats.outliers_influence import variance_inflation_factor

# scaler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC

# pipeline
from sklearn.pipeline import Pipeline 

# resampling
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

# model selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold, KFold, cross_val_score

# scoring
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# pca
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager
# 한글설정(MAC)
f_path = '/Library/Fonts/NanumGothic.ttf'
font_manager.FontProperties(fname=f_path).get_name()
from matplotlib import rc
rc('font', family = 'NanumGothic')

# 한글 설정 (WIN)
# from matplotlib import rc
# plt.rcParams['axes.unicode_minus'] = False
# f_path= "C:/Windows/Fonts/malgun.ttf"
# font_name= font_manager.FontProperties(fname=f_path).get_name()
# rc('font', family =font_name)
# plt.rc('font', family='Malgun Gothic')

pd.options.display.float_format = '{:.5f}'.format
```

### 1.1 데이터 불러오기
- 본 프로젝트는 쏘카로부터 데이터를 제공받아 진행된 프로젝트입니다. 
```python
# 1. 데이터 불러오기 
socar_df = pd.read_csv("0. raw_data/insurance_fraud_detect_data.csv")
pd.set_option('display.max_columns', len(socar_df.columns))
socar = socar_df.copy()
```
## EDA

### 1. SweetViz
```python
socar_tr = socar_df[socar_df["test_set"] == 0]
socar_test = socar_df[socar_df["test_set"] == 1]
socar_report = sv.compare([socar_tr, "Train"], [socar_test, "Test"], "fraud_YN")
socar_report.show_html('./socar_report.html')
```
<img src="https://user-images.githubusercontent.com/71831714/104716672-97831700-576b-11eb-80e5-867e81d60082.png"></img>


### 2. Seaborn

#### 1) 불균형한 데이터 분포
```python3
sns.countplot('fraud_YN', data=socar_df)
plt.title("Fraud Distributions \n", fontsize=14)
plt.show()
```
???????????????????????????????????????????????????????
<img src="https://user-images.githubusercontent.com/71831714/104717802-3b20f700-576d-11eb-9e68-13a5fbfa24ff.png"></img>

#### 2) 컬럼별 분포도 확인
```python3
var = socar.columns.values

t0 = socar.loc[socar['fraud_YN']==0]
t1 = socar.loc[socar['fraud_YN']==1]

sns.set_style('whitegrid')
plt.figure()
fig,ax = plt.subplots(7,4,figsize=(16,28))


for i, feature in enumerate(var):
    plt.subplot(7,4,i+1)
    sns.kdeplot(t0[feature], bw=0.5, label = 'fraud_0')
    sns.kdeplot(t1[feature], bw=0.5, label = 'fraud_1')

    plt.xlabel(feature,fontsize=12)
    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which = 'major', labelsize=12)

plt.show()
```
<img src="https://user-images.githubusercontent.com/71831714/104717879-5b50b600-576d-11eb-9417-b8a123987454.png"></img>

#### 3) 상관관계 히트맵
```python3
mask = np.zeros_like(socar.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(socar.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
```
<img src="https://user-images.githubusercontent.com/71831714/104718021-9652e980-576d-11eb-868f-03c3c7843e5e.png"></img>

#### 4) 다중공선성
```python3
pd.DataFrame({"VIF Factor": [variance_inflation_factor(socar.values, idx) 
                             for idx in range(socar.shape[1])], "features": socar.columns})
```
<img src="https://user-images.githubusercontent.com/71831714/104718280-fea1cb00-576d-11eb-9ed3-d63b4d36eec4.png"></img>

#### 5) 변수 관찰
```python3
def make_graph(column):
    fig,ax = plt.subplots(2, 2, figsize=(20,12))

    t0 = socar[socar['fraud_YN']==0]
    t1 = socar[socar['fraud_YN']==1]


    plt.subplot(2,2,1)
    ax0 = sns.countplot(column, data=socar[socar['fraud_YN']==0])
    for p in ax0.patches:
        count = p.get_height()
        x = p.get_x() 
        y = p.get_y() + p.get_height()
        ax0.annotate(count, (x, y))
    plt.title("non-fraud {}".format(column))    

    plt.subplot(2,2,2)
    ax1 = sns.countplot(column, data=socar[socar['fraud_YN']==1])
    for p in ax1.patches:
        count = p.get_height()
        x = p.get_x() + 0.3
        y = p.get_y() + p.get_height()
        ax1.annotate(count, (x, y))
    plt.title("fraud {}".format(column))    

    plt.subplot(2,2,3)
    socar_df[socar_df['fraud_YN']==1][column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("fraud {}".format(column))    

    plt.subplot(2,2,4)
    sns.kdeplot(t0[column], bw=0.5, label = 'non-fraud')
    sns.kdeplot(t1[column], bw=0.5, label = 'fraud')
    plt.title("fraud vs non-fraud")   

    plt.show()

make_graph('accident_hour')
```
<img src="https://user-images.githubusercontent.com/71831714/104718424-33158700-576e-11eb-83f4-57562b70928a.png"></img>

## Preprocessing 
### 스케일링 
- standard scailing 
- log scailing 
- minmax scailing 
- robust scailing 

### 샘플링
imbalanced data 처리를 위한 다양한 샘플링 기법 시도 
- random under 
- random over 
- smote
- adasyn 
- smotenn

### PCA
- 차원 축소 기법을 통한 데이터 노이즈 제거 

### 원핫인코딩 
- 적용안함 
- 일부 카테고리 변수 
- 모든 카테고리 변수 

### 결측치처리 / 이상치 제거 
- 평균값/중앙값/최빈값/KNN imputer 를 활용한 결측치 보간 진행 

## 하이퍼파라미터 튜닝 
### 모델별 최적성능을 위해 아래와 같이 파라미터 튜닝 작업을 시도 

## Modeling  
### 모델 학습 
- logistic regression
- decision tree
- random forest 
- lgbm
- svm 

## Model evaluation 
### 모델 성능 평가 (metrics)
- 정확도와 재현률을 기준으로 성능 평가 진행 

## Conclusion
- 

## 함께한 분석가 :thumbsup:
  
- 김미정 
  - 
  - GitHub: 
  
- 김성준
  - 
  - GitHub: 
  
- 이정려
  - 
  - GitHub: 
  
- 전예나
  - 
  - GitHub: 
