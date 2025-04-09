# -*- coding: utf-8 -*-
"""ML_token_feature_2_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vvLyUXIRFHQ0bICGtmZqmk0c1CAY5fVt

# トークンを特徴量にした機械学習による文書分類（回帰問題）
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor  # Random Forestのクラス
from xgboost import XGBRegressor  # XGBoostのクラス
from lightgbm import LGBMRegressor  # LightGBMのクラス
from transformers import AutoTokenizer
from typing import Tuple
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# データセットの読み込み
train_df = pd.read_csv('/content/llm-class/dataset/train.csv')
valid_df = pd.read_csv('/content/llm-class/dataset/validation.csv')
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# データセットを結合
all_sentences = train_dataset['sentence'] + valid_dataset['sentence']
all_labels = train_dataset['label'] + valid_dataset['label']

# トークナイズと特徴量化
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in all_sentences]
tokenized_sentences = [' '.join(tokens) for tokens in tokenized_sentences]

# 特徴量の作成
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X = vectorizer.fit_transform(tokenized_sentences)

# トレーニングデータとバリデーションデータの分割
num_train_samples = len(train_dataset)
X_train = X[:num_train_samples]
X_valid = X[num_train_samples:]
train_labels = all_labels[:num_train_samples]
valid_labels = all_labels[num_train_samples:]

# Random Forestモデルの訓練（回帰）
regressor_rf = RandomForestRegressor()
regressor_rf.fit(X_train, train_labels)

# バリデーションデータで予測（回帰）
valid_predictions_rf = regressor_rf.predict(X_valid)

# 回帰向けのメトリクス計算
mse_rf = mean_squared_error(valid_labels, valid_predictions_rf)
mae_rf = mean_absolute_error(valid_labels, valid_predictions_rf)
rmse_rf = np.sqrt(mean_squared_error(valid_labels, valid_predictions_rf))

print("■RandomForest Regression")
print("MSE:", mse_rf)
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

# 予測結果のDataFrameを作成
predictions_df_rf = pd.DataFrame({
    'label': valid_labels,
    'sentence': valid_dataset['sentence'],
    'predicted_value': valid_predictions_rf
})

# 予測結果をCSVに保存
predictions_df_rf.to_csv("/content/llm-class/results/regression/results_rf.csv", index=False)

"""### （参考） XGBoost"""

# XGBoostモデルの訓練（回帰）
regressor_xgb = XGBRegressor()
regressor_xgb.fit(X_train, train_labels)

# バリデーションデータで予測（回帰）
valid_predictions_xgb = regressor_xgb.predict(X_valid)

# 回帰向けのメトリクス計算
mse_xgb = mean_squared_error(valid_labels, valid_predictions_xgb)
mae_xgb = mean_absolute_error(valid_labels, valid_predictions_xgb)
rmse_xgb = np.sqrt(mean_squared_error(valid_labels, valid_predictions_xgb))

print("■XGBoost Regression")
print("MSE:", mse_xgb)
print("MAE:", mae_xgb)
print("RMSE:", rmse_xgb)


# 予測結果のDataFrameを作成
predictions_df_rf = pd.DataFrame({
    'label': valid_labels,
    'sentence': valid_dataset['sentence'],
    'predicted_value': valid_predictions_rf
})

# 予測結果をCSVに保存
predictions_df_rf.to_csv("/content/llm-class/results/regression/results_xgb.csv", index=False)


"""###  （参考）  LightGBM"""

params = {
    'verbose': -1  # 警告メッセージを非表示にする
}

regressor_lgbm= LGBMRegressor(**params)
regressor_lgbm.fit(X_train, train_labels)

# バリデーションデータで予測（回帰）
valid_predictions_lgbm = regressor_lgbm.predict(X_valid)

# 回帰向けのメトリクス計算
mse_lgbm = mean_squared_error(valid_labels, valid_predictions_lgbm)
mae_lgbm = mean_absolute_error(valid_labels, valid_predictions_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(valid_labels, valid_predictions_lgbm))

print("■LightGBM Regression")
print("MSE:", mse_lgbm)
print("MAE:", mae_lgbm)
print("RMSE:", rmse_lgbm)

# 予測結果のDataFrameを作成
predictions_df_rf = pd.DataFrame({
    'label': valid_labels,
    'sentence': valid_dataset['sentence'],
    'predicted_value': valid_predictions_rf
})

# 予測結果をCSVに保存
predictions_df_rf.to_csv("/content/llm-class/results/regression/results_lgbm.csv", index=False)


