cancer project

TCGA 데이터에서 26개의 암 종류 데이터 전처리 

-- muation이 있고 없음 즉 0,1 데이터로 모든 유전자x환자 행렬 변환
1. SVD로 행렬 변환 후 만든 transformer 모델에 투입 score 0.29
   문제: 데이터셋의 암의 아종에 따라 데이터가 불균형해서 문제가 발생
2. random forest 실제 score  0.59

3. soft voting system 도입
   Losistic, randomForest, CatBoost, LGBMClassdier  test F1 0.76, Acc 0.76 실제 score 0.60

4. PudMedBert(pretrained)를 fine Tuning down task로 암 아종 분류 사용
  token이 제한되어 있어 주요한 암 돌연변이 유형을 우선순위로 두고, 100개를 추출함
  prompt는 the patient has the following mutation: {gene mutationType}. what is the most likely cancer type?
  PubMedBert는 PubMed에서 제공하는 논문 abstract와 open access full text를 학습에 사용 
  GAD(Genetic Association Database) Tesk에 가장 높은 성능을 보여(83.96) 사용해 보았음 
