# 패키지 설치가 안 되어 있다면 아래 주석을 해제하고 실행하세요
# install.packages("correctR")
library(correctR)

# 1. 데이터를 1차원 벡터로 평탄화 (Flatten)
# 파이썬의 중첩 리스트 순서에 맞춰 나열합니다. (Seed 1의 Fold 1~5 -> Seed 2 -> Seed 3)
ctmp_gin_auc <- c(
  0.95539, 0.95488, 0.95517, 0.95493, 0.95506, # Seed 1 (Fold 1~5)
  0.95365, 0.95374, 0.95369, 0.95379, 0.95378, # Seed 2 (Fold 1~5)
  0.95386, 0.95431, 0.95346, 0.95443, 0.95389  # Seed 3 (Fold 1~5)
)

gin_auc <- c(
  0.95339, 0.95356, 0.95344, 0.95351, 0.95322,
  0.95336, 0.95325, 0.95343, 0.95312, 0.95321,
  0.95361, 0.95321, 0.95367, 0.95318, 0.95368
)

a3tgcn_auc <- c(
  0.95235, 0.95210, 0.95286, 0.95268, 0.95268,
  0.95223, 0.95247, 0.95241, 0.95247, 0.95285,
  0.95257, 0.95310, 0.95301, 0.95296, 0.95264
)

# 2. 반복(k, r) 인덱스 생성
# k (Fold 번호): 1~5가 3번 반복됨 (1 2 3 4 5 1 2 3 4 5 1 2 3 4 5)
k_seq <- rep(1:5, times = 3)
# r (Seed 번호): 1, 2, 3이 각각 5번씩 반복됨 (1 1 1 1 1 2 2 2 2 2 3 3 3 3 3)
r_seq <- rep(1:3, each = 5)

# 3. correctR 규격에 맞는 데이터프레임 구성
# CTMP_GIN vs GIN 데이터
df_gin <- data.frame(
  model = rep(c("CTMP_GIN", "GIN"), each = 15),
  values = c(ctmp_gin_auc, gin_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs A3TGCN 데이터
df_a3tgcn <- data.frame(
  model = rep(c("CTMP_GIN", "A3TGCN"), each = 15),
  values = c(ctmp_gin_auc, a3tgcn_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# 4. 통계 검정 파라미터 설정 (TEDS 데이터셋 기준)
n1 <- 948014  # Train 세트 사이즈
n2 <- 209121  # Test 세트 사이즈
k_folds <- 5   # Fold 개수
r_repeats <- 3 # Seed(반복) 개수

# 5. Repeated K-Fold Corrected T-Test 실행
cat("=== CTMP_GIN vs GIN ===\n")
res_gin <- repkfold_ttest(data = df_gin, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_gin)

cat("\n=== CTMP_GIN vs A3TGCN ===\n")
res_a3tgcn <- repkfold_ttest(data = df_a3tgcn, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_a3tgcn)
