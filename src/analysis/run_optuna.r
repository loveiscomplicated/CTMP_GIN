# 패키지 설치가 안 되어 있다면 아래 주석을 해제하고 실행하세요
# install.packages("correctR")
library(correctR)

# ============================================================
# 데이터 출처: wandb_export_2026-04-27T10_22_22_952_09_00.csv
# 모든 값은 test_auc 컬럼 기준
# 순서: Seed 3 (Fold 4→0), Seed 2 (Fold 4→0), Seed 1 (Fold 4→0)
# ============================================================

# 1. 데이터를 1차원 벡터로 평탄화 (Flatten)
ctmp_gin_auc <- c(
  0.95490, 0.95496, 0.95570, 0.95547, 0.95507, # Seed 3 (Fold 4~0)
  0.95500, 0.95503, 0.95458, 0.95508, 0.95500, # Seed 2 (Fold 4~0)
  0.95537, 0.95519, 0.95507, 0.95526, 0.95490  # Seed 1 (Fold 4~0)
)

gin_auc <- c(
  0.95277, 0.95281, 0.95274, 0.95300, 0.95287, # Seed 3 (Fold 4~0)
  0.95323, 0.95286, 0.95258, 0.95305, 0.95292, # Seed 2 (Fold 4~0)
  0.95339, 0.95348, 0.95378, 0.95359, 0.95379  # Seed 1 (Fold 4~0)
)

a3tgcn_auc <- c(
  0.95466, 0.95438, 0.95429, 0.95460, 0.95488, # Seed 3 (Fold 4~0)
  0.95375, 0.95420, 0.95418, 0.95390, 0.95384, # Seed 2 (Fold 4~0)
  0.95479, 0.95304, 0.95499, 0.95522, 0.95474  # Seed 1 (Fold 4~0)
)

# ctmp_gin_remove_gate (Gated Fusion 제거 ablation)
no_gate_auc <- c(
  0.95415, 0.95373, 0.95402, 0.95457, 0.95406, # Seed 3 (Fold 4~0)
  0.95455, 0.95513, 0.95508, 0.95531, 0.95505, # Seed 2 (Fold 4~0)
  0.95567, 0.95505, 0.95559, 0.95594, 0.95591  # Seed 1 (Fold 4~0)
)

# ctmp_gin_no_preprocessing (전처리 제거 ablation)
no_process_auc <- c(
  0.95436, 0.95395, 0.95393, 0.95431, 0.95390, # Seed 3 (Fold 4~0)
  0.95469, 0.95469, 0.95485, 0.95496, 0.95482, # Seed 2 (Fold 4~0)
  0.95546, 0.95511, 0.95550, 0.95544, 0.95541  # Seed 1 (Fold 4~0)
)

# ctmp_gin_fully_connected_edge (완전 연결 엣지 ablation)
fully_auc <- c(
  0.95469, 0.95477, 0.95491, 0.95483, 0.95509, # Seed 3 (Fold 4~0)
  0.95435, 0.95429, 0.95468, 0.95456, 0.95440, # Seed 2 (Fold 4~0)
  0.95466, 0.95524, 0.95522, 0.95501, 0.95506  # Seed 1 (Fold 4~0)
)

# [신규] gingru 모델 (CSV에 새로 추가된 baseline)
gingru_auc <- c(
  0.95273, 0.95296, 0.95243, 0.95287, 0.95300, # Seed 3 (Fold 4~0)
  0.95171, 0.95138, 0.95168, 0.95141, 0.95218, # Seed 2 (Fold 4~0)
  0.95183, 0.95199, 0.95217, 0.95201, 0.95203  # Seed 1 (Fold 4~0)
)


# 2. 반복(k, r) 인덱스 생성
# k (Fold 번호): 1~5가 3번 반복됨
k_seq <- rep(1:5, times = 3)
# r (Seed 번호): 1, 2, 3이 각각 5번씩 반복됨
r_seq <- rep(1:3, each = 5)

# 3. correctR 규격에 맞는 데이터프레임 구성
# CTMP_GIN vs GIN
df_gin <- data.frame(
  model = rep(c("CTMP_GIN", "GIN"), each = 15),
  values = c(ctmp_gin_auc, gin_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs A3TGCN
df_a3tgcn <- data.frame(
  model = rep(c("CTMP_GIN", "A3TGCN"), each = 15),
  values = c(ctmp_gin_auc, a3tgcn_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs Ablation - no_gate
df_no_gate <- data.frame(
  model = rep(c("CTMP_GIN", "no_gate"), each = 15),
  values = c(ctmp_gin_auc, no_gate_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs Ablation - no_preprocessing
df_no_preprocess <- data.frame(
  model = rep(c("CTMP_GIN", "no_preprocessing"), each = 15),
  values = c(ctmp_gin_auc, no_process_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# CTMP_GIN vs Ablation - fully_connected_edge
df_fully <- data.frame(
  model = rep(c("CTMP_GIN", "fully_connected_edge"), each = 15),
  values = c(ctmp_gin_auc, fully_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)

# [신규] CTMP_GIN vs GINGRU
df_gingru <- data.frame(
  model = rep(c("CTMP_GIN", "GINGRU"), each = 15),
  values = c(ctmp_gin_auc, gingru_auc),
  k = rep(k_seq, times = 2),
  r = rep(r_seq, times = 2)
)


# ============================================================
# [참고] 원본 코드에 있던 ctmp_gin_auc_2, ctmp_gin_auc_3 관련
# 데이터프레임/검정은 CSV 파일에 해당 실험 결과가 없어
# 정의가 불가능하므로 제외했습니다.
# 추후 데이터가 추가되면 동일한 패턴으로 확장 가능합니다.
# ============================================================


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

cat("\n=== CTMP_GIN vs GINGRU ===\n")
res_gingru <- repkfold_ttest(data = df_gingru, n1 = n1, n2 = n2, k = k_folds, r = r_repeats, tailed = "two")
print(res_gingru)