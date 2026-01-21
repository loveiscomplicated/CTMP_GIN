import os
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np

CURDIR = os.path.dirname(__file__)
PARDIR = os.path.join(CURDIR, '..')
root = os.path.join(PARDIR, 'data_tensor_cache')
data_path = os.path.join(root, 'raw', 'missing_corrected.csv')

def train_xgboost(train_idx, test_idx, df, logger):
    """
    XGBoost 모델을 학습하고 평가하는 메인 함수
    """
    # 1. 데이터 불러오기
    data = pd.read_csv(data_path)
    X = data.drop(columns=['REASONb'])
    y = data['REASONb']

    # 2. (중요) 범주형 변수의 데이터 타입을 'category'로 변경
    # enable_categorical=True를 사용하기 위한 필수 단계입니다.
    # 모든 피처를 범주형으로 가정합니다.
    print("Converting feature dtypes to 'category'...")
    X = X.astype('category')
    print("Data types converted.")

    # 3. 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 4. 최종 모델 학습 및 X_test로 평가
    print("\n--- Training final model and evaluating with X_test ---")

    final_xgb_model = XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        random_state=42
    )
    final_xgb_model.fit(X_train, y_train)

    # X_test에 대한 예측 수행
    y_pred_proba = final_xgb_model.predict_proba(X_test)[:, 1]
    y_pred = final_xgb_model.predict(X_test)

    # 평가 지표 계산
    test_logloss = log_loss(y_test, y_pred_proba)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = final_xgb_model.score(X_test, y_test)
    test_f1_score = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    print(f"Final Test Log Loss (BCE): {test_logloss:.4f}")
    print(f"Final Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test F1-Score: {test_f1_score:.4f}")
    print(f"Final Test Precision: {test_precision:.4f}")
    print(f"Final Test Recall: {test_recall:.4f}")



    # 5. 최종 모델의 피처 중요도 시각화
    print("\n--- Feature Importance (Gain) ---")
    feature_names = X_train.columns
    importances = final_xgb_model.feature_importances_
    feature_series = pd.Series(importances, index=feature_names)
    sorted_importances = feature_series.sort_values(ascending=False)

    print(sorted_importances)

    plt.figure(figsize=(12, 8))
    sorted_importances.plot(kind='bar')
    plt.title('Feature Importance (Gain)')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()