# 쏘카 보험 사기 탐지 머신러닝 프로젝트
![image](https://user-images.githubusercontent.com/72847093/104838734-3476b900-5900-11eb-9428-96d19d7840d8.png)

## 개요 

### 1.1  프로젝트 주제
- 보험금을 목적으로 한 사기 집단의 렌터카를 이용한 사기 수법이 기승 
- 이에 쏘카의 사고 데이터를 통해 Fraud 유저에 대한 예측을 실현하고자 함 

### 1.2 프로젝트 진행순서
1. 개요 
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
pip install statsmodels
```

## Setting 

### 1.1 환경설정
```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

pd.options.display.float_format = '{:.5f}'.format
```

### 1.1 데이터 불러오기
- 본 프로젝트는 쏘카로부터 데이터를 제공받아 진행된 프로젝트입니다. 
## 개요 
#### 문제 해결 솔루션 
- 클래스의 분포가 과도하게 불균형하여 샘플링을 활용한 문제 해결 시도
- 모델의 성능을 향상시키기 위한 다양한 데이터 전처리 진행
- 데이터의 노이즈를 줄일 수 있는 차원축소 기법을 사용 

## EDA

#### 1) 불균형한 데이터 분포
![image](https://user-images.githubusercontent.com/72847093/105167733-6d12ce80-5b5c-11eb-82a0-d72ad3437618.png)
#### 2) 컬럼별 분포도 확인
![image](https://user-images.githubusercontent.com/72847093/105167827-90d61480-5b5c-11eb-88b6-db9c7f6031e6.png)
#### 3) 컬럼별 데이터 분석 
![image](https://user-images.githubusercontent.com/72847093/105167875-a21f2100-5b5c-11eb-9b79-6284ac729281.png)
![image](https://user-images.githubusercontent.com/72847093/105167921-b3682d80-5b5c-11eb-9a73-3b3c15742a47.png)
![image](https://user-images.githubusercontent.com/72847093/105167945-c0851c80-5b5c-11eb-9055-534a4283abb3.png)
#### 4) 상관관계 히트맵
![image](https://user-images.githubusercontent.com/72847093/105167986-ced33880-5b5c-11eb-96fc-9a4889157231.png)

## Preprocessing

### 결측치처리 / 이상치 제거 
- 평균값/중앙값/최빈값/KNN imputer 를 활용한 결측치 보간 진행 

### PCA
- 차원 축소 기법을 통한 데이터 노이즈 제거 

### 샘플링
imbalanced data 처리를 위한 다양한 샘플링 기법 시도 
- random under 
- random over 
- smote
- adasyn 
- smotenn

### 스케일링 
- standard scailing 
- log scailing 
- minmax scailing 
- robust scailing 

### 원핫인코딩 
- 적용안함 
- 일부 카테고리 변수 
- 모든 카테고리 변수 

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
  - GitHub: 
  
- 김성준
  - GitHub: 
  
- 이정려
  - GitHub: 
  
- 전예나
  - GitHub: 
