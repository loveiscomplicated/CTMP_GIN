# 패키지 설치가 안 되어 있다면 아래 주석을 해제하고 실행하세요
# install.packages("correctR")
library(correctR)

# 1. 데이터를 1차원 벡터로 평탄화 (Flatten)
# 모든 모델 동일 순서: Seed 3의 Fold 4→0, Seed 2의 Fold 4→0, Seed 1의 Fold 4→0
# (ctmp_gin의 wandb 기록 순서에 맞춰 통일)
ctmp_gin_auc <- c(
  0.95539, 0.95488, 0.95517, 0.95493, 0.95506, # Seed 3 (Fold 4~0)
  0.95365, 0.95374, 0.95369, 0.95379, 0.95378, # Seed 2 (Fold 4~0)
  0.95386, 0.95431, 0.95346, 0.95443, 0.95389  # Seed 1 (Fold 4~0)
)

# [수정] gin의 seed 순서를 ctmp_gin과 동일하게 맞춤 (기존: Seed 1→3→2, 수정: Seed 3→2→1)
gin_auc <- c(
  0.95336, 0.95325, 0.95343, 0.95312, 0.95321, # Seed 3 (Fold 4~0)
  0.95361, 0.95321, 0.95367, 0.95318, 0.95368, # Seed 2 (Fold 4~0)
  0.95339, 0.95356, 0.95344, 0.95351, 0.95322  # Seed 1 (Fold 4~0)
)

a3tgcn_auc <- c(
  0.95235, 0.95210, 0.95286, 0.95268, 0.95268, # Seed 3 (Fold 4~0)
  0.95223, 0.95247, 0.95241, 0.95247, 0.95285, # Seed 2 (Fold 4~0)
  0.95257, 0.95310, 0.95301, 0.95296, 0.95264  # Seed 1 (Fold 4~0)
)

# [수정] wandb CSV 최신 값으로 업데이트
no_gate_auc <- c(
  0.95415, 0.95286, 0.95387, 0.95302, 0.95286, # Seed 3 (Fold 4~0)
  0.95274, 0.95337, 0.95352, 0.95375, 0.95340, # Seed 2 (Fold 4~0)
  0.95230, 0.95337, 0.95218, 0.95284, 0.95275  # Seed 1 (Fold 4~0)
)

# [수정] wandb CSV 최신 값으로 업데이트
no_process_auc <- c(
  0.95491, 0.95484, 0.95412, 0.95499, 0.95484, # Seed 3 (Fold 4~0)
  0.95428, 0.95418, 0.95451, 0.95400, 0.95426, # Seed 2 (Fold 4~0)
  0.95495, 0.95494, 0.95530, 0.95523, 0.95513  # Seed 1 (Fold 4~0)
)

# [수정] wandb CSV 최신 값으로 업데이트
fully_auc <- c(
  0.95514, 0.95451, 0.95493, 0.95463, 0.95451, # Seed 3 (Fold 4~0)
  0.95412, 0.95385, 0.95431, 0.95343, 0.95404, # Seed 2 (Fold 4~0)
  0.95535, 0.95482, 0.95485, 0.95520, 0.95492  # Seed 1 (Fold 4~0)
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

# CTMP_GIN vs Ablation 데이터 - no_gate
df_no_gate <- data.frame(
  model = rep(c("CTMP_GIN", "no_gate"), each = 15),
  values = c(ctmp_gin_auc, no_gate_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs Ablation 데이터 - no_process (no missing value imputation)
df_no_preprocess <- data.frame(
  model = rep(c("CTMP_GIN", "no_preprocessing"), each = 15),
  values = c(ctmp_gin_auc, no_process_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs Ablation 데이터 - fully_connected_edge
df_fully <- data.frame(
  model = rep(c("CTMP_GIN", "fully_connected_edge"), each = 15),
  values = c(ctmp_gin_auc, fully_auc),
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

cat("\n=== CTMP_GIN vs Gated_Fusion removed ===\n")
res_no_gate <- repkfold_ttest(data = df_no_gate, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_no_gate)

cat("\n=== CTMP_GIN vs Preprocessing method removed ===\n")
res_no_preprocess <- repkfold_ttest(data = df_no_preprocess, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_no_preprocess)

cat("\n=== CTMP_GIN vs Fully connected edge ===\n")
res_fully <- repkfold_ttest(data = df_fully, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_fully)
