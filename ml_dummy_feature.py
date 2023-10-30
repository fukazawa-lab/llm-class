
"""### ダミー変数化"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple
from sklearn.metrics import precision_score, recall_score

# データを読み込む
train_data = pd.read_csv('train_num.csv')
validation_data = pd.read_csv('validation_num.csv')

# 特徴量を選択
categorical_features = ['天候', '路面状態', '道路形状', '信号機', '車道幅員', '道路線形', '衝突地点']

# 訓練データとバリデーションデータを結合
combined_data = pd.concat([train_data, validation_data], ignore_index=True)

# 訓練データとバリデーションデータを含めてダミー変数化
combined_data = pd.get_dummies(combined_data, columns=categorical_features, drop_first=True)

# 訓練データとバリデーションデータに再分割
X_train = combined_data[:len(train_data)]
X_val = combined_data[len(train_data):]

# 説明変数からtargetを削除
X_train = X_train.drop(columns=['target'])
X_val = X_val.drop(columns=['target'])

# 目的変数を分離
y_train = train_data['target']
y_val = validation_data['target']

# モデルを訓練する
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# バリデーションデータで予測を行う
y_pred = clf.predict(X_val)

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.sign(predictions)
    precision = precision_score(labels, predictions, average='macro')  # または average='micro' など適切なオプションを選択してください
    recall = recall_score(labels, predictions, average='macro')  # または average='micro' など適切なオプションを選択してください
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy, "precision": precision, "recall": recall}

metrics_dict = compute_metrics((y_pred, y_val))
accuracy = metrics_dict["accuracy"]
precision = metrics_dict["precision"]
recall = metrics_dict["recall"]

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

