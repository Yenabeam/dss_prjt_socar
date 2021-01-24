# 쏘카 보험 사기 탐지 머신러닝 프로젝트
###### 발표자료 URL : https://github.com/Yenabeam/dss_prjt_socar/raw/master/PPTPrjt_ML_socar_YENAJEON_%EC%B5%9C%EC%A2%85_%EB%A7%88%EC%8A%A4%ED%82%B9.pdf
![image](https://user-images.githubusercontent.com/72847093/104838734-3476b900-5900-11eb-9428-96d19d7840d8.png)

## 1. 개요 

### 1.1  프로젝트 주제
- 2020년 상반기 보험사기의 적발금액은 규모 약 4526억원으로, 코로나19의 유행과 경기 침체로 인한 생계형 보험 사기와 함께 매년 증가 추세 
- 보험 사기 수법 중 하나인 공유차량을 이용한 ‘뒷쿵’과 같은 사기 수법에 주목하여,  쏘카의 차량사고 데이터를 활용한 보험금을 목적으로 한 렌터카 사고 사기 집단에 대한 분류 예측을 하고자 본 프로젝트를 진행 
### 1.2 데이터 
- 본 데이터는 쏘카로부터 제공받은 데이터로 일부 feature 대한 정보 비공개로 수정하였음 

## 2. 솔루션 
### 2.1 Fraud 유저가 34개인 불균형 데이터를 학습과 모델에 적용
![image](https://user-images.githubusercontent.com/72847093/105628573-09044900-5e81-11eb-8391-586dc1f54f3e.png)![image](https://user-images.githubusercontent.com/72847093/105628559-f25df200-5e80-11eb-99e9-0ae51710403f.png)
- 사고 유저 중 정상 집단과 사기 집단의 데이터값이 과도하게 불균형한 모습을 확인 
- 다양한 데이터 샘플링 기법을 사용하여 불균형 데이터를 처리함 
- 사용 샘플링 기법 : random under sampling, smote, adasyn, random over, smote-enn, smote-tomek

### 2.2 데이터 전처리 
- 모델링 전 데이터에 대한 다양한 전처리 작업을 시도함 
- 결측치  : ‘알수없음’, ‘확인불가’에 대한 데이터 결측치 처리 진행 
- 스케일링 : standard scailing, minmax scailing, log scailing, robust scailing 
- 인코딩 : 원핫인코딩

### 2.3 데이터 노이즈
- 데이터의 상세 값을 확인 시 ‘알수없음’, ‘확인불가’에 대한 내용이 확인되어 데이터를 잘 표현하는 주성분으로 차원을 축소를 시도함 

![image](https://user-images.githubusercontent.com/72847093/105631271-0d386280-5e91-11eb-84aa-aae4594e8f38.png)![image](https://user-images.githubusercontent.com/72847093/105628651-73b58480-5e81-11eb-8a31-21743cc7b33f.png)


## 3. 성능 평가 
- Recall, accuracy 를 주요 성능 평가 지표 
- 본 모델을 통한 기업 영업 손실 방지를 목적으로, 실제 사기 집단의 예측을 목적으로 하여, Recall 수치를 성능 평가 기준으로 삼기로 하였음 
- 모델이 모든 예측 값을 사기 집단으로 판단하는 것을 방지하고자 accuracy를 참고 성능 평가 요소로 확인하였음 
- Best#2 모델의 경우 최종 정확도 0.5 / 재현률 0.57의 성능을 확인

## 4. 구조  
![image](https://user-images.githubusercontent.com/72847093/105631057-cd24b000-5e8f-11eb-87dd-c3b79e3e277e.png)

## 5. Gist 
#### 1. 샘플링의 사용
#### 2. 알고리즘의 선택 
#### 3. Feature 선택 

## 6. 참고 논문
- '불균형 이분 데이터 분류분석을 위한 데이터 마이닝절차' 
http://www.webmail.databaser.net/moniwiki/pds/DataAnalysis/mining_process_01.pdf

## 함께한 분석가 :thumbsup:
- 김미정 
  - GitHub: https://github.com/LeilaYK
  
- 김성준
  - GitHub: https://github.com/alltimeno1
  
- 이정려
  - GitHub: https://github.com/jungryo
  
- 전예나 : EDA 진행, 다양한 샘플링 적용을 통한 모델링, 결측치 처리, 발표 진행 및 PPT/리드미 작성 
  - GitHub: https://github.com/Yenabeam

###### 본 프로젝트는 패스트캠퍼스 데이터사이언스 취업스쿨 15th 머신러닝 프로젝트로 진행되었습니다.
