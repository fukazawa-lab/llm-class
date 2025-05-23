# -*- coding: utf-8 -*-
"""ML_numerical_feature_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1duUrANwlfPBhvbhPoVukCivqkQt_xVt1

## 構造化データの回帰問題

### 数値特徴量
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # RandomForestRegressorへ変更
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 回帰のためのメトリクスを追加
from typing import Tuple

# データを読み込む
train_data = pd.read_csv('/content/llm-class/dataset/train_num.csv')
validation_data = pd.read_csv('/content/llm-class/dataset/validation_num.csv')

# 説明変数と目的変数を分離する
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_val = validation_data.drop(columns=['label'])
y_val = validation_data['label']

# モデルを訓練する（RandomForestRegressorに変更）
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# バリデーションデータで予測を行う
y_pred = regressor.predict(X_val)

# 回帰向けのメトリクス計算
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mse)

print("■RandomForest Regression")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# 予測結果のDataFrameを作成
predictions_df = pd.DataFrame({
    'label': y_val,
    'predicted_label': y_pred,
    **X_val.to_dict('series')
})

# 予測結果をCSVに保存
predictions_df.to_csv("/content/llm-class/results/regression/results_num.csv", index=False)
