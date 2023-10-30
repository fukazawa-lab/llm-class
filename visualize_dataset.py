"""# （参考） データセット統計の可視化"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import japanize_matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from datasets import Dataset, ClassLabel
from datasets import load_dataset
import numpy as np
import pandas as pd

plt.rcParams["font.size"] = 12  # 文字サイズを大きくする

def visualize_text_length(dataset: Dataset):
    # モデル名を指定してトークナイザを読み込む
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """データセット中のテキストのトークン数の分布をグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[length] += 1
    # length_counterの値から棒グラフを描画する
    # グラフのサイズを設定
    plt.figure(figsize=(4, 3))  # 幅 8 インチ、高さ 6 インチ

    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()


from collections import Counter
import matplotlib.pyplot as plt

def visualize_labels(dataset):
    """データセット中のラベル分布をグラフとして描画"""
    # データセット中のラベルの数を数える
    label_counter = Counter()
    for data in dataset:
        label_id = data["label"]
        label_counter[label_id] += 1

    # ラベルIDをラベル名に変換するための辞書
    label_id_to_name = {0: "ラベル0", 1: "ラベル1"}

    # グラフのサイズを設定
    plt.figure(figsize=(4, 3))  # 幅 8 インチ、高さ 6 インチ

    # label_counterを棒グラフとして描画する
    label_names = [label_id_to_name[label_id] for label_id in label_counter.keys()]
    label_counts = list(label_counter.values())
    plt.bar(label_names, label_counts, width=0.6)
    plt.xlabel("ラベル")
    plt.ylabel("事例数")
    plt.show()

visualize_text_length(train_dataset)
visualize_text_length(valid_dataset)

visualize_labels(train_dataset)
visualize_labels(valid_dataset)
