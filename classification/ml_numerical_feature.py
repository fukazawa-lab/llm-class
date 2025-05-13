import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack

# データの読み込み
train_data = pd.read_csv('/content/llm-class/dataset/train_num.csv')
validation_data = pd.read_csv('/content/llm-class/dataset/validation_num.csv')

# 説明変数と目的変数を分離
X_train_df = train_data.drop(columns=['label'])
y_train = train_data['label']
X_val_df = validation_data.drop(columns=['label'])
y_val = validation_data['label']

# 全ラベルを取得してエンコード
all_labels = pd.concat([y_train, y_val], ignore_index=True)
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
all_labels_enc = label_encoder.transform(all_labels)

# DataFrameをsparse matrixに変換
X_train = csr_matrix(X_train_df.values)
X_val = csr_matrix(X_val_df.values)

# 学習データに存在しないラベルをチェックし、平均ベクトルで補完
missing_labels = set(all_labels_enc) - set(y_train_enc)

for label in missing_labels:
    y_train_enc = np.append(y_train_enc, label)

    # 平均特徴量を1行ベクトルにして追加
    mean_features = X_train.mean(axis=0)  # 平均を求める (1, n_features)
    mean_csr = csr_matrix(mean_features)  # csr_matrixに変換
    X_train = vstack([X_train, mean_csr])  # 縦方向に追加

# モデル学習
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_enc)

# バリデーションデータで予測
pred_proba = clf.predict_proba(X_val)
pred_labels = np.argmax(pred_proba, axis=1)
pred_labels_orig = label_encoder.inverse_transform(pred_labels)

# 結果をDataFrameにまとめて保存
predictions_df = pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_labels_orig,
    **X_val_df.to_dict('series')
})
predictions_df.to_csv('/content/llm-class/results/classification/result_num.csv', index=False)

# 混同行列の保存
conf_matrix = confusion_matrix(y_val, pred_labels_orig, labels=label_encoder.classes_)
conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
conf_matrix_df.to_csv('/content/llm-class/results/classification/confusion_matrix_num.csv')

# 評価指標の出力
print("■RandomForest")
print("Accuracy:", accuracy_score(y_val, pred_labels_orig))
print("Precision:", precision_score(y_val, pred_labels_orig, average='weighted'))
print("Recall:", recall_score(y_val, pred_labels_orig, average='weighted'))
