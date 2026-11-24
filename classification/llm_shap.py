        raise FileNotFoundError(f'{MODEL_DIR} がありません。先に train_bert() を実行してください')

    valid = _data('validation.csv')
    texts = valid.sentence.fillna('').astype(str).iloc[:samples].tolist()
    if not texts:
        raise ValueError('検証文章がありません')

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    if not 0 <= target_class < model.config.num_labels:
        raise ValueError('target_classがクラス範囲外です')

    # BERTによる予測確率
    def predict(batch):
        batch = [str(t) for t in batch]
        outputs = []
        for start in range(0, len(batch), batch_size):
            enc = tokenizer(
                batch[start:start + batch_size],
                padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                outputs.append(
                    torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
                )
        return np.concatenate(outputs, axis=0)

    # Slow日本語Tokenizerにoffset_mappingを補完
    explainer = shap.Explainer(
        predict,
        shap.maskers.Text(_SlowJapaneseTokenizerWithOffsets(tokenizer)),
        output_names=[str(i) for i in range(model.config.num_labels)],
        algorithm='partition'
    )

    sv = explainer(texts, max_evals=max_evals)
    # print('SHAP解析件数:', len(texts), '対象クラス:', target_class)
    shap.plots.text(sv[0, :, target_class])

    # サブワード単位のSHAP値を取得
    rows = []
    for i in range(len(texts)):
        vals = np.asarray(sv.values[i])
        if vals.ndim == 2:
            vals = vals[:, target_class]

        for token, value in zip(sv.data[i], vals):
            token = str(token).strip()
            if token and np.isfinite(value):
                rows.append({
                    'sample_id': i,
                    'token': token,
                    'shap_value': float(value)
                })

    tokens = pd.DataFrame(rows, columns=['sample_id', 'token', 'shap_value'])
    OUT.mkdir(parents=True, exist_ok=True)
    tokens.to_csv(OUT / 'bert_shap_tokens.csv', index=False, encoding='utf-8-sig')

    # SHAP表示用フィルタ（学習・予測には影響しない）
    def keep(token):
        w = str(token).strip().removeprefix('##')
        return (
            bool(w)
            and (keep_numbers or not any(c.isdigit() for c in w))
            and (keep_hiragana or not re.fullmatch(r'[\u3040-\u309F]+', w))
            and (keep_symbols or not all(ud.category(c)[0] in 'PS' for c in w))
        )

    tokens = tokens[tokens.token.map(keep)].copy()
    if tokens.empty:
        raise ValueError('フィルタ後のトークンがありません。フィルタ条件を確認してください')

    # 実際に出現したトークンだけで平均を計算
    summary = tokens.groupby('token')['shap_value'].agg(
        mean_shap='mean',
        mean_abs_shap=lambda x: x.abs().mean(),
        count='count'
    ).sort_values('mean_abs_shap', ascending=False)

    summary.to_csv(OUT / 'bert_shap_global.csv', encoding='utf-8-sig')
    # print(summary.head(top_n).to_string())

    # SHAP図：不存在トークンの0点は表示しない
    plot = summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(plot))))
    rng = np.random.default_rng(SEED)

    for y, token in enumerate(plot.index):
        values = tokens.loc[
            tokens.token == token, 'shap_value'
        ].to_numpy()

        ax.scatter(
            values,
            y + rng.uniform(-0.15, 0.15, len(values)),
            c='#FF0051', alpha=0.6, s=25
        )
        ax.scatter(
            values.mean(), y,
            marker='D', c='black', s=65, zorder=5
        )

    ax.set_yticks(range(len(plot)), labels=plot.index)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set(
        xlabel=f'SHAP value (class {target_class})',
        ylabel='Subword',
        title='BERT SHAP: observed subwords only (black diamond = mean)'
    )

    fig.tight_layout()
    fig.savefig(OUT / 'bert_shap_subwords.png', dpi=160, bbox_inches='tight')
    plt.show()

    return summary
