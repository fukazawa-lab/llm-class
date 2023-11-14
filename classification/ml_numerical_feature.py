# -*- coding: utf-8 -*-
"""ML_numerical_feature_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1duUrANwlfPBhvbhPoVukCivqkQt_xVt1

## 構造化データの分類問題

### 数値特徴量
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple
from sklearn.metrics import precision_score, recall_score

# データを読み込む
train_data = pd.read_csv('/content/llm-class/dataset/classification/train_num.csv')
validation_data = pd.read_csv('/content/llm-class/dataset/classification/validation_num.csv')

# 説明変数と目的変数を分離する
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']
X_val = validation_data.drop(columns=['target'])
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

