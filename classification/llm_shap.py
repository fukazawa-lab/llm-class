# -*- coding: utf-8 -*-
"""交通事故分類演習：RF（表／文章）、BERT、およびSHAP。
Colabから importlib.reload(llm) して関数を呼び出す想定。
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')
os.environ.setdefault('WANDB_DISABLED', 'false')
from pathlib import Path
import re
import inspect
import numpy as np
import pandas as pd

ROOT = Path('llm-class')
DATA = ROOT / 'dataset'
OUT = ROOT / 'results' / 'classification'
MODEL_DIR = OUT / 'bert_finetuned'
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v3'
SEED = 42


def _data(name):
    return pd.read_csv(DATA / name)


def _report(y, pred, filename):
    from sklearn.metrics import classification_report, confusion_matrix
    OUT.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y) | set(pred))
    print(classification_report(y, pred, zero_division=0))
    pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels).to_csv(OUT / f'{filename}_confusion_matrix.csv')
    pd.DataFrame({'label': np.asarray(y), 'predicted_label': np.asarray(pred)}).to_csv(OUT / f'{filename}_predictions.csv', index=False)


def _rf_shap(model, x_valid, names, prefix, shap_samples=100, top_n=25, target_class=1, dependence_top_n=12):
    import shap
    import matplotlib.pyplot as plt
    n = min(shap_samples, len(x_valid))
    if n < 1:
        raise ValueError('SHAP対象データがありません')
    x = x_valid[:n] if isinstance(x_valid, np.ndarray) else x_valid.iloc[:n]
    class_idx = list(model.classes_).index(target_class) if target_class in model.classes_ else 0
    raw = shap.TreeExplainer(model).shap_values(x)
    if isinstance(raw, list):
        values = np.asarray(raw[class_idx])
    else:
        raw = np.asarray(raw)
        values = raw[:, :, class_idx] if raw.ndim == 3 else raw
    print(f'SHAP対象クラス: {model.classes_[class_idx]} / {n}件')
    shap.summary_plot(values, x, feature_names=list(names), max_display=top_n)
    importance = pd.Series(np.abs(values).mean(axis=0), index=names, name='mean_abs_shap').sort_values(ascending=False)
    display(importance.head(top_n).to_frame()) if 'display' in globals() else print(importance.head(top_n))
    importance.to_csv(OUT / f'{prefix}_shap_global.csv')
    pd.DataFrame(values, columns=names).to_csv(OUT / f'{prefix}_shap_values.csv', index=False)
    # 添付rf_shap.pyに準じた依存プロット：特徴量値 vs SHAP値
    import math
    from matplotlib.colors import Normalize
    features = importance.head(min(dependence_top_n, len(importance))).index
    if len(features):
        cols, rows = 2, math.ceil(len(features) / 2)
        fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows), squeeze=False)
        feature_index = {name: i for i, name in enumerate(names)}
        for ax, feature in zip(axes.flat, features):
            j = feature_index[feature]
            feature_values = np.asarray(x.iloc[:, j] if hasattr(x, 'iloc') else x[:, j], dtype=float)
            ax.scatter(feature_values, values[:, j], c=feature_values,
                       cmap=shap.plots.colors.red_blue, s=17, alpha=0.7,
                       edgecolors='none')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set(title=str(feature), xlabel='Feature value', ylabel='SHAP value')
        for ax in axes.flat[len(features):]:
            ax.axis('off')
        fig.suptitle(f'{prefix}: SHAP dependence (class {model.classes_[class_idx]})', y=1.002)
        fig.tight_layout()
        fig.savefig(OUT / f'{prefix}_shap_dependence.png', dpi=160, bbox_inches='tight')
        plt.show()
    return importance


def rf_table(shap_samples=100, top_n=25, target_class=1, dependence_top_n=12):
    """元の表形式の説明変数でRFを学習しSHAPを表示。"""
    from sklearn.ensemble import RandomForestClassifier
    train, valid = _data('train_num.csv'), _data('validation_num.csv')
    for part in (train, valid):
        if 'label' not in part:
            raise ValueError('label列が必要です')
    x_train = pd.get_dummies(train.drop(columns='label'), dummy_na=True)
    x_valid = pd.get_dummies(valid.drop(columns='label'), dummy_na=True).reindex(columns=x_train.columns, fill_value=0)
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians).fillna(0)
    x_valid = x_valid.fillna(medians).fillna(0)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(x_train, train.label)
    _report(valid.label, model.predict(x_valid), 'rf_table')
    return _rf_shap(model, x_valid, x_train.columns, 'rf_table', shap_samples, top_n, target_class, dependence_top_n)


def rf_text(shap_samples=100, top_n=25, target_class=1, dependence_top_n=12):
    """文章をTF-IDF化してRFを学習しSHAPを表示。"""
    from transformers import AutoTokenizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    train, valid = _data('train.csv'), _data('validation.csv')
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    vectorizer = TfidfVectorizer(tokenizer=lambda s: [t for t in tok.tokenize(str(s)) if t not in tok.all_special_tokens],
                                 token_pattern=None, lowercase=False, max_features=1000)
    x_train = vectorizer.fit_transform(train.sentence.fillna('')).toarray()
    x_valid = vectorizer.transform(valid.sentence.fillna('')).toarray()
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(x_train, train.label)
    _report(valid.label, model.predict(x_valid), 'rf_text')
    return _rf_shap(model, x_valid, vectorizer.get_feature_names_out(), 'rf_text', shap_samples, top_n, target_class, dependence_top_n)


def train_bert(epochs=5, batch_size=8, learning_rate=2e-5):
    """日本語BERTをファインチューニングし、SHAP用にモデルを保存。"""
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    set_seed(SEED)
    train, valid = _data('train.csv'), _data('validation.csv')
    if not {'label', 'sentence'} <= set(train.columns) or not {'label', 'sentence'} <= set(valid.columns):
        raise ValueError('train.csvとvalidation.csvにlabel,sentence列が必要です')
    classes = sorted(set(train.label))
    if classes != list(range(len(classes))):
        raise ValueError('BERTのlabelは0から始まる連続整数にしてください')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def encode(batch):
        result = tokenizer(batch['sentence'], truncation=True, max_length=512)
        result['labels'] = batch['label']
        return result
    train_ds = Dataset.from_pandas(train[['sentence','label']].fillna({'sentence':''}), preserve_index=False)
    valid_ds = Dataset.from_pandas(valid[['sentence','label']].fillna({'sentence':''}), preserve_index=False)
    train_ds = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(encode, batched=True, remove_columns=valid_ds.column_names)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes))
    params = inspect.signature(TrainingArguments.__init__).parameters
    settings = dict(output_dir=str(OUT / 'checkpoints'), per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size, learning_rate=learning_rate,
                    num_train_epochs=epochs, logging_strategy='epoch', save_strategy='epoch',
                    load_best_model_at_end=True, metric_for_best_model='f1',
                    report_to='none', fp16=torch.cuda.is_available())
    if 'eval_strategy' in params:
        settings['eval_strategy'] = 'epoch'
    elif 'evaluation_strategy' in params:
        settings['evaluation_strategy'] = 'epoch'
    else:
        raise RuntimeError('TrainingArgumentsに評価設定が見つかりません')
    if 'warmup_ratio' in params:
        settings['warmup_ratio'] = 0.1
    if 'save_safetensors' in params:
        settings['save_safetensors'] = False
    settings = {k:v for k,v in settings.items() if k in params}
    args = TrainingArguments(**settings)
    def metrics(p):
        logits, labels = p
        pred = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro', zero_division=0)
        return {'accuracy': accuracy_score(labels, pred), 'precision': precision, 'recall': recall, 'f1': f1}
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    pred = trainer.predict(valid_ds).predictions.argmax(axis=1)
    _report(valid.label, pred, 'bert')
    print('モデル保存先:', MODEL_DIR)
    return trainer


def _is_content_token(token, keep_numbers=True):
    pattern = r'[一-龯々ァ-ヶーA-Za-z0-9]' if keep_numbers else r'[一-龯々ァ-ヶーA-Za-z]'
    return bool(re.search(pattern, str(token).strip()))


def bert_shap(samples=30, max_evals=300, target_class=1, top_n=30,
              batch_size=8, keep_numbers=True):
    """BERTのSHAP解析：文章別の点（赤）と平均（黒◆）。"""
    import torch
    import shap
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    valid = _data("validation.csv")
    texts = valid.sentence.fillna("").astype(str).iloc[:samples].tolist()
    if not texts:
        raise ValueError("検証文章がありません")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def predict(batch):
        outputs = []
        for i in range(0, len(batch), batch_size):
            enc = tokenizer(list(batch[i:i+batch_size]), padding=True,
                            truncation=True, max_length=512,
                            return_tensors="pt").to(device)
            with torch.no_grad():
                outputs.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(outputs)

    explainer = shap.Explainer(
        predict, shap.maskers.Text(tokenizer), algorithm="partition"
    )
    sv = explainer(texts, max_evals=max_evals)
    shap.plots.text(sv[0, :, target_class])

    rows = []
    for i in range(len(texts)):
        values = np.asarray(sv.values[i])
        if values.ndim == 2:
            values = values[:, target_class]
        for token, value in zip(sv.data[i], values):
            token = str(token).strip()
            if token and np.isfinite(value):
                rows.append((i, token, float(value)))

    tokens = pd.DataFrame(rows, columns=["sample_id", "token", "shap_value"])
    OUT.mkdir(parents=True, exist_ok=True)
    tokens.to_csv(OUT / "bert_shap_tokens.csv", index=False, encoding="utf-8-sig")

    tokens = tokens[tokens.token.map(
        lambda t: _is_content_token(t, keep_numbers)
    )]
    matrix = tokens.pivot_table(
        index="sample_id", columns="token", values="shap_value",
        aggfunc="sum", fill_value=0
    ).reindex(range(len(texts)), fill_value=0)

    summary = pd.DataFrame({
        "global_mean_shap": matrix.mean(),
        "global_mean_abs_shap": matrix.abs().mean(),
        "count": tokens.groupby("token").size()
    })
    summary = summary.reindex(
        summary.global_mean_shap.abs().sort_values(ascending=False).index
    )
    summary.to_csv(OUT / "bert_shap_global.csv", encoding="utf-8-sig")
    print(summary.head(top_n).to_string())

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.36)))
    rng = np.random.default_rng(42)

    for y, token in enumerate(summary.head(top_n).index[::-1]):
        values = matrix[token].to_numpy()
        ax.scatter(values, y + rng.uniform(-0.15, 0.15, len(values)),
                   c="#FF0051", alpha=0.5, s=24)
        ax.scatter(values.mean(), y, marker="D", c="black", s=75, zorder=5)

    ax.set_yticks(range(min(top_n, len(summary))))
    ax.set_yticklabels(summary.head(top_n).index[::-1])
    ax.axvline(0, color="gray", lw=1)
    ax.set(xlabel=f"SHAP Value (class {target_class})",
           ylabel="Token", title="BERT SHAP: Individual Samples and Mean")
    plt.tight_layout()
    plt.show()
    return summary
