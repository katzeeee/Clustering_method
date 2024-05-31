## Clustering method

여러 clustering 기법 중 많이 사용하는 알고리즘 몇 가지를 정리한 repo




### K-MEANS
- unsupervised learning에 속하는 알고리즘으로 데이터를 k개의 군집으로 묶는 기법
  - 군집이란 비슷한 특성을 지닌 데이터들을 모아놓은 그룹

- 데이터의 평균(means)를 활용하여 클러스터링하고, 평균은 클러스터의 중심과 데이터간 차이의 평균을 뜻함

군집화를 진행하는 순서
  1. 군집의 개수 정하기:

       데이터를 준비한 뒤 먼저 군집의 개수를 정해야 하는데, 이는 사람이 정해야 함

       군집의 개수를 어떻게 정하냐에 따라 결과가 크게 달라져 군집의 개수를 정하는
       방법론으로 몇 가지 방법들이 존재한다고 함
     
         1. Rule of Thumb
     
         2. Elbow Method
     
         3. Information Criterion Approach

  2. 중심점(Centroid) 설정 :

       k개의 군집을 설정 했으면 k개의 중심점을 설정해야 함

       초기 중심점도 군집의 개수와 같이 어떤 값을 선택하는가에 따라 성능이 크게 달라져
       마찬가지로 몇가지 방법론이 존재
     
         1. Randomly select
     
         2. Manually assign
     
         3. K-means++(실제로 자주 사용되는 초기화 방법)
     

  3. 데이터 군집 할당 :

       군집과 중심점을 정했으면 우선 데이터들을 중심점과의 거리(유클리드 거리)를 비교하여
       군집을 형성

  4. 데이터 군집 할당 :

       초기 설정한 중심점을 군집 내의 데이터들의 중간(평균)으로 재설정

  5. 데이터 군집 할당 :

       중심점을 갱신 했으므로 다시 데이터를 군집에 배정하고 중심점 이동이 없을 때 까지 반복




### DBSCAN(Density Based Spatial Clustering of Applications with Noise)
- 거리 개념이 아닌 데이터의 밀도 차이를 파악하는 밀도 기반 클러스터링 기법

- 특징:
  - 알고리즘이 데이터 밀도 차이를 자동으로 감지하며 군집을 생성하므로 사용자가 군집 개수를 지정할 수 없음
  - DBSCAN은 데이터 밀도가 자주 변하거나, 아예 모든 데이터의 밀도가 크게 변하지 않으면 군집화 성능이 떨어짐
  - Feature의 개수가 많으면 군집화 성능이 떨어짐

- 입실론 주변 영역 내에 포함되는 최소 데이터 개수를 충족시키는가 아닌가에 따라 데이터 포인트를 아래와 같이 정의
  - 핵심 포인트 (Core Point): 주변 영역 내에 최소 데이터 개수 이상의 타 데이터를 가지고 있을 경우 해당 데이터
  - 이웃 포인트 (Neighbor Point): 주변 영역 내에 위치한 타 데이터
  - 경계 포인트 (Border Point): 주변 영역 내에 최소 데이터 개수 이상의 이웃 포인트를 가지고 있지 않지만, 핵심 포인트를 이웃 포인트로 가지고 있는 데이터
  - 잡음 포인트 (Noise Point): 최소 데이터 개수 이상의 이웃 포인트를 가지고 있지 않으며, 핵심 포인트도 이웃 포인트로 가지고 있지 않는 데이터
 
- epsilon, min_samples를 하이퍼 파라미터로 가짐
  - epsilon : 개별 데이터를 중심으로 입실론 반경의 원형의 영역을 가짐. 즉 원의 반지름
  - min_samples : 원형의 영역 안에 min_samples 이상의 데이터가 포함되면 핵심 포인트




### Mean Shift Clustering

- K-means는 중심점이 데이터간 거리의 평균점으로 이동을 하지만, Mean-Shift는 중심점을 데이터가 가장 많이 모여 있는곳, 즉 밀도가 가장 높은 곳으로 이동

- 특징:
  - KDE (Kernel Density Estimation)로 확률 밀도 함수를 찾음
  - KDE를 이용하여 데이터 포인트들이 데이터 분포가 높은 곳으로 이동하면서 군집화를 수행
  - 별도의 군집화 개수를 지정하지 않고, 데이터 분포도에 기반하여 자동으로 군집 개수 선정

- 군집화 순서
  1. 개별 데이터의 특정 반경 내에 주변 데이터를 포함한 데이터 분포도를 KDE 이용하여 계산
  2. KDE로 계산된 데이터 분포도가 높은 방향으로 데이터 이동
  3. 모든 데이터에 1 - 2를 수행하면서 데이터를 이동, 개별 데이터들이 군집 중심점에 모임
  4. 지정된 반복만큼 전체 데이터에 대해 KDE 기반으로 데이터를 이동시키면서 군집화 수행
  5. 개별 데이터들이 모인 중심점을 군집 중심점으로 설정

- Bandwidth가 커지면 개별 커널 함수의 영향력이 작아져 KDE 그래프가 부드러워지고, 반대로 작아지면 그래프가 뾰족해짐. sklearn의 estimate_banwidth()를 사용하면 최적 bandwidth를 계산 해 줌


- Bandwidth를 하이퍼 파라미터로 가짐
  - epsilon : 개별 데이터를 중심으로 입실론 반경의 원형의 영역을 가짐. 즉 원의 반지름
  - min_samples : 원형의 영역 안에 min_samples 이상의 데이터가 포함되면 핵심 포인트


### Gaussian Mixture Model
- K-Means는 특정 중심점을 기반으로 거리적으로 퍼져있는 데이터 세트에 군집화를 적용하면 효율적인데 그 반대는 비효율적임
![Untitled (8)](https://github.com/katzeeee/Clustering_method/assets/104491909/a6ef0a8f-58aa-4bea-9495-d9317fb33668)

- 특징:
  - 알고리즘이 데이터 밀도 차이를 자동으로 감지하며 군집을 생성하므로 사용자가 군집 개수를 지정할 수 없음
  - DBSCAN은 데이터 밀도가 자주 변하거나, 아예 모든 데이터의 밀도가 크게 변하지 않으면 군집화 성능이 떨어짐
  - Feature의 개수가 많으면 군집화 성능이 떨어짐

- 입실론 주변 영역 내에 포함되는 최소 데이터 개수를 충족시키는가 아닌가에 따라 데이터 포인트를 아래와 같이 정의
  - 핵심 포인트 (Core Point): 주변 영역 내에 최소 데이터 개수 이상의 타 데이터를 가지고 있을 경우 해당 데이터
  - 이웃 포인트 (Neighbor Point): 주변 영역 내에 위치한 타 데이터
  - 경계 포인트 (Border Point): 주변 영역 내에 최소 데이터 개수 이상의 이웃 포인트를 가지고 있지 않지만, 핵심 포인트를 이웃 포인트로 가지고 있는 데이터
  - 잡음 포인트 (Noise Point): 최소 데이터 개수 이상의 이웃 포인트를 가지고 있지 않으며, 핵심 포인트도 이웃 포인트로 가지고 있지 않는 데이터
 
- epsilon, min_samples를 하이퍼 파라미터로 가짐
  - epsilon : 개별 데이터를 중심으로 입실론 반경의 원형의 영역을 가짐. 즉 원의 반지름
  - min_samples : 원형의 영역 안에 min_samples 이상의 데이터가 포함되면 핵심 포인트
