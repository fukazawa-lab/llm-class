import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack

# =========================
# データ読み込み
# =========================
train_data = pd.read_csv('llm-class/dataset/train_num.csv')
validation_data = pd.read_csv('llm-class/dataset/validation_num.csv')

X_train_df = train_data.drop(columns=['label']).astype(np.float32)
y_train = train_data['label']

X_val_df = validation_data.drop(columns=['label']).astype(np.float32)
y_val = validation_data['label']

# =========================
# ラベルエンコード
# =========================
all_labels = pd.concat([y_train, y_val], ignore_index=True)

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
all_labels_enc = label_encoder.transform(all_labels)

# =========================
# sparse変換（float32統一）
# =========================
X_train = csr_matrix(X_train_df.to_numpy(dtype=np.float32))
X_val = csr_matrix(X_val_df.to_numpy(dtype=np.float32))

# =========================
# missing label補完（元コード踏襲）
# =========================
missing_labels = set(all_labels_enc) - set(y_train_enc)

for label in missing_labels:
    y_train_enc = np.append(y_train_enc, label)

    mean_features = X_train.mean(axis=0)
    mean_csr = csr_matrix(mean_features)

    X_train = vstack([X_train, mean_csr])

# =========================================================
# 評価関数
# =========================================================
def evaluate(name, y_true, y_pred):
    print(f"\n■{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))


# =========================================================
# ① RandomForest
# =========================================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train_enc)

pred = rf.predict(X_val)
pred_labels_orig = label_encoder.inverse_transform(pred)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_labels_orig,
}).to_csv('llm-class/results/classification/result_num_rf.csv', index=False)

evaluate("RandomForest", y_val, pred_labels_orig)


# =========================================================
# ② XGBoost
# =========================================================
xgb = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

xgb.fit(X_train, y_train_enc)

pred = xgb.predict(X_val)
pred_labels_orig = label_encoder.inverse_transform(pred)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_labels_orig,
}).to_csv('llm-class/results/classification/result_num_xgb.csv', index=False)

evaluate("XGBoost", y_val, pred_labels_orig)


# =========================================================
# ③ LightGBM
# =========================================================
lgb = LGBMClassifier(
    verbose=-1,
    random_state=42
)

lgb.fit(X_train, y_train_enc)

pred = lgb.predict(X_val)
pred_labels_orig = label_encoder.inverse_transform(pred)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_labels_orig,
}).to_csv('llm-class/results/classification/result_num_lgb.csv', index=False)

evaluate("LightGBM", y_val, pred_labels_orig)