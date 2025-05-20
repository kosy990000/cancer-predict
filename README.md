cancer project

TCGA 데이터에서 26개의 암 종류 데이터 전처리 

-- muation이 있고 없음 즉 0,1 데이터로 모든 유전자x환자 행렬 변환
1. SVD로 행렬 변환 후 만든 transformer 모델에 투입 score 0.29
  -> 문제 데이터셋의 암의 아종에 따라 불균형해서 문제가 발생
2. random forest 실제 score  0.59

3. soft voting system 도입
   Losistic, randomForest, CatBoost, LGBMClassdier 내부 test F1 0.76, Acc 0.76 실제 score 0.60
