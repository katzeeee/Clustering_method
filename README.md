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
