# -*- coding: utf-8 -*-
"""ML_token_feature_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vvLyUXIRFHQ0bICGtmZqmk0c1CAY5fVt

# トークンを特徴量にした機械学習による文書分類
"""
from sklearn.metrics import precision_score, recall_score

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from typing import Tuple
from lightgbm import LGBMClassifier  # LightGBMのクラス
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier  # Random Forestのクラス
from datasets import load_dataset, Dataset, ClassLabel
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Tuple

# CSVファイルからデータを読み込む
train_df = pd.read_csv('/content/llm-class/dataset/classification/train.csv')
valid_df = pd.read_csv('/content/llm-class/dataset/classification/validation.csv')
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

# Create a LabelEncoder and fit it to your class labels
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
encoded_labels_train = label_encoder.transform(train_labels)
encoded_labels_valid = label_encoder.transform(valid_labels)


# all_labels_encodedに含まれていて、encoded_labels_trainに含まれていないラベルを見つける
missing_labels = set(all_labels_encoded) - set(encoded_labels_train)

# encoded_labels_trainの末尾に不足しているラベルを追加
for label in missing_labels:
    encoded_labels_train = np.append(encoded_labels_train, label)
    # 不足しているラベルに対応する特徴量を求め、X_trainに追加
    missing_label_indices = np.where(all_labels_encoded == label)[0]
    missing_label_features = csr_matrix.mean(X[missing_label_indices], axis=0)
    X_train = vstack([X_train, missing_label_features])

# Random Forestモデルの訓練
clf = RandomForestClassifier()
clf.fit(X_train, encoded_labels_train)

# バリデーションデータで予測
valid_predictions = clf.predict_proba(X_valid)

# 確率の最も高いクラスを取得
predicted_labels = np.argmax(valid_predictions, axis=1)

# LabelEncoderを使用して予測値を元のクラスラベルに逆変換
original_valid_predictions = label_encoder.inverse_transform(predicted_labels)

# 正解率、Precision、Recallの計算
conf_matrix = confusion_matrix(label_encoder.transform(valid_labels), predicted_labels)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
precision = precision_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')
recall = recall_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')

print("■RandomForest")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 予測結果をCSVに出力
predictions_df  = pd.DataFrame({
    'label': valid_labels,
    'predicted_label': original_valid_predictions,
    'sentence': valid_dataset['sentence']
})
predictions_df.to_csv('/content/llm-class/results/classification/result_rf.csv', index=False)

# 実際のラベルと予測されたラベルから混合行列を計算
conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])

# ユニークなラベルのリストを取得
unique_labels = sorted(set(predictions_df['label'].unique()) | set(predictions_df['predicted_label'].unique()))

# 混合行列をCSVファイルとして保存
conf_matrix_df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
conf_matrix_df.to_csv('/content/llm-class/results/classification/confusion_matrix_rf.csv')



"""### （参考） XGBoost"""

# XGBoostモデルの訓練
clf = XGBClassifier()
clf.fit(X_train, encoded_labels_train)

# バリデーションデータで予測
valid_predictions = clf.predict_proba(X_valid)

# 確率の最も高いクラスを取得
predicted_labels = np.argmax(valid_predictions, axis=1)

# LabelEncoderを使用して予測値を元のクラスラベルに逆変換
original_valid_predictions = label_encoder.inverse_transform(predicted_labels)

# 正解率、Precision、Recallの計算
conf_matrix = confusion_matrix(label_encoder.transform(valid_labels), predicted_labels)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
precision = precision_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')
recall = recall_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')

print("■XGBoost")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 予測結果をCSVに出力
predictions_df  = pd.DataFrame({
    'label': valid_labels,
    'predicted_label': original_valid_predictions,
    'sentence': valid_dataset['sentence']
})
predictions_df.to_csv('/content/llm-class/results/classification/result_xgb.csv', index=False)

# 実際のラベルと予測されたラベルから混合行列を計算
conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])

# ユニークなラベルのリストを取得
unique_labels = sorted(set(predictions_df['label'].unique()) | set(predictions_df['predicted_label'].unique()))

# 混合行列をCSVファイルとして保存
conf_matrix_df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
conf_matrix_df.to_csv('/content/llm-class/results/classification/confusion_matrix_xgb.csv')



"""###  （参考）  LightGBM"""

# ハイパーパラメータの設定
params = {
    'objective': 'multiclass',  # 分類の場合は'multiclass'を指定
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_data_in_leaf': 50,
    'max_depth': -1,
    'bagging_fraction': 0.8,
    'num_class': len(np.unique(encoded_labels_train)),  # クラスの数
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'metric': 'multi_logloss',  # ロジスティック損失を使用
    'min_split_gain': 0.1,  # これを調整してみてください
    'feature_fraction': 0.8,  # 特徴のサブサンプリングを試してみてください
    'verbose': -1  # 警告メッセージを非表示にする
}
# LightGBMモデルの訓練
clf = LGBMClassifier(**params)
clf.fit(X_train, encoded_labels_train)

# バリデーションデータで予測
valid_predictions = clf.predict_proba(X_valid)


# 確率の最も高いクラスを取得
predicted_labels = np.argmax(valid_predictions, axis=1)

# LabelEncoderを使用して予測値を元のクラスラベルに逆変換
original_valid_predictions = label_encoder.inverse_transform(predicted_labels)

# 正解率、Precision、Recallの計算
conf_matrix = confusion_matrix(label_encoder.transform(valid_labels), predicted_labels)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
precision = precision_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')
recall = recall_score(label_encoder.transform(valid_labels), predicted_labels, average='weighted')

print("■LightGBM")
# 混合行列の計算
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)



# 予測結果をCSVに出力
predictions_df  = pd.DataFrame({
    'label': valid_labels,
    'predicted_label': original_valid_predictions,
    'sentence': valid_dataset['sentence']
})
predictions_df.to_csv('/content/llm-class/results/classification/result_lgb.csv', index=False)

# 実際のラベルと予測されたラベルから混合行列を計算
conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])

# ユニークなラベルのリストを取得
unique_labels = sorted(set(predictions_df['label'].unique()) | set(predictions_df['predicted_label'].unique()))

# 混合行列をCSVファイルとして保存
conf_matrix_df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
conf_matrix_df.to_csv('/content/llm-class/results/classification/confusion_matrix_lgb.csv')

