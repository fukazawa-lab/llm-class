# -*- coding: utf-8 -*-
"""LLM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z0xw36QwD6kCAxLzfnJ3DXR2Cv-evNoW

# 大規模言語モデルのファインチューニング

# 1 環境の準備
"""
import torch
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pprint import pprint
from datasets import Dataset, ClassLabel
from datasets import load_dataset
import pandas as pd
from typing import Union
from transformers import BatchEncoding

# 乱数シードを42に固定
set_seed(42)
print("乱数シード設定完了")

"""# 2 データセットの準備"""

# Hugging Face Hub上のllm-book/wrime-sentimentのリポジトリからデータを読み込む
# original_train_dataset = load_dataset("llm-book/wrime-sentiment", split="train")
# valid_dataset = load_dataset("llm-book/wrime-sentiment", split="validation")

# original_train_dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split="train")
# valid_dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split="validation")

# 学習データからN個のデータだけを抽出
# train_dataset = original_train_dataset.shuffle(seed=42).select([i for i in range(1000)])

# CSVファイルからデータを読み込む
original_train_df = pd.read_csv('/content/llm-class/dataset/classification/train.csv')
valid_df = pd.read_csv('/content/llm-class/dataset/classification/validation.csv')
train_dataset = Dataset.from_pandas(original_train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# pprintで見やすく表示する
pprint(train_dataset[0])

print("")

"""# 3. トークン化"""

# モデル名を指定してトークナイザを読み込む
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# トークナイザのクラス名を確認
print(type(tokenizer).__name__)

# テキストのトークン化
tokens = tokenizer.tokenize(train_dataset[0]['sentence'])
print(tokens)

# データのトークン化

def preprocess_text_classification(
    example: dict[str, str | int]
) -> BatchEncoding:
    """文書分類の事例のテキストをトークナイズし、IDに変換"""
    encoded_example = tokenizer(example["sentence"], max_length=512)
    # モデルの入力引数である"labels"をキーとして格納する
    encoded_example["labels"] = example["label"]
    return encoded_example

encoded_train_dataset = train_dataset.map(
    preprocess_text_classification,
    remove_columns=train_dataset.column_names,
)
encoded_valid_dataset = valid_dataset.map(
    preprocess_text_classification,
    remove_columns=valid_dataset.column_names,
)

# トークン化の確認
print(encoded_train_dataset[0])

"""# 4 ミニバッチ構築"""

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# ミニバッチ結果の確認
batch_inputs = data_collator(encoded_train_dataset[0:4])
pprint({name: tensor.size() for name, tensor in batch_inputs.items()})

"""# 5 モデルの準備"""

from transformers import AutoModelForSequenceClassification
from collections import Counter

# ファインチューニング済みモデルを読み込む
# model_name = "fine_tuned_model_directory"  # モデルが保存されたディレクトリパス


# データセットからラベルの一覧を取得
labels = [example["label"] for example in train_dataset]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ラベルの数を計算
num_labels = np.max(labels) + 1

label2id = {label: id for id, label in enumerate(range(num_labels))}
id2label = {id: label for id, label in enumerate(range(num_labels))}
model = (AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    label2id=label2id,  # ラベル名からIDへの対応を指定
    id2label=id2label,  # IDからラベル名への対応を指定
).to(device))


model = (AutoModelForSequenceClassification
    .from_pretrained(model_name, num_labels=num_labels)
    .to(device))
    

"""# 6 訓練の実行"""

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output_wrime",  # 結果の保存フォルダ
    per_device_train_batch_size=8,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=8,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type="linear",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=2,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
    fp16=True,  # 自動混合精度演算の有効化
)

def compute_accuracy(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    # predictionsは各ラベルについてのスコア
    # 最もスコアの高いインデックスを予測ラベルとする
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')  # または average='micro' など適切なオプションを選択してください
    recall = recall_score(labels, predictions, average='macro')  # または average='micro' など適切なオプションを選択してください

    return {"accuracy": accuracy, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_accuracy,
)
trainer.train()

# # # ファインチューニング済みモデルを保存
# model.save_pretrained("fine_tuned_model_directory")

# 予測結果の取得
predictions = trainer.predict(encoded_valid_dataset)

# 通常は0番目のラベルに対応する予測値
predictions_df = pd.DataFrame({
    'label': predictions.label_ids,
    'predicted_label': predictions.predictions.argmax(axis=1),
    'sentence': valid_dataset["sentence"]
})

predictions_df.to_csv("/content/llm-class/dataset/classification/results.csv", index=False)

"""# 7 精度検証"""

print("■評価結果")
# 検証セットでモデルを評価
eval_metrics = trainer.evaluate(encoded_valid_dataset)
accuracy = eval_metrics['eval_accuracy']
precision = eval_metrics['eval_precision']
recall = eval_metrics['eval_recall']

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

from sklearn.metrics import confusion_matrix

# is_correctを初期化
is_correct = (predictions.predictions.argmax(axis=1) == predictions.label_ids)
# 混合行列の計算
cm = confusion_matrix(predictions.label_ids, predictions.predictions.argmax(axis=1))

print(cm)
