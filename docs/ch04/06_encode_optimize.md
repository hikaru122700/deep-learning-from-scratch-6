# エンコード最適化

## 学習目標

BPEエンコードを**効率的に実装**する方法を理解する。

## 主要概念

### 1. 問題点

元のエンコード実装は、すべてのマージルールを順番に適用：
- O(num_merges) 回のマージ操作
- 多くは該当するペアがない

### 2. 優先度ベースのエンコード

```python
def _encode_text(self, text):
    ids = list(text.encode("utf-8"))

    def get_merge_priority(pair):
        return self.merge_rules.get(pair, float('inf'))

    while len(ids) > 1:
        # 現在のペアを取得
        counts = count_pairs(ids)

        # 最優先ペア（最も早く学習されたペア）を特定
        best_pair = min(counts, key=get_merge_priority)

        # マージ可能か確認
        if best_pair not in self.merge_rules:
            break

        # マージを実行
        new_id = self.merge_rules[best_pair]
        ids = merge(ids, best_pair, new_id)

    return ids
```

### 3. 優先度の考え方

マージルールは学習順に格納されているため：
- 早く学習されたペア = 頻出 = 優先度が高い
- `merge_rules.get(pair, float('inf'))` で未登録ペアは最低優先度

### 4. 完全なエンコード処理

```python
def encode(self, input_text, show_progress=False):
    pattern = '(' + re.escape(self.end_token) + ')'
    texts = re.split(pattern, input_text)
    all_ids = []

    texts = tqdm(texts) if show_progress else texts

    for text in texts:
        if text == self.end_token:
            all_ids.append(self.end_token_id)
        else:
            for pretoken in pretokenize_iter(text):
                ids = self._encode_text(pretoken)
                all_ids.extend(ids)

    return all_ids
```

## アルゴリズムの違い

| 方式 | 計算量 | 特徴 |
|------|--------|------|
| 全ルール適用 | O(num_merges × n) | 単純だが遅い |
| 優先度ベース | O(k × n) | k はマージ回数 |

## ポイント

1. **min で最優先を選択**: 学習順序が小さい = 優先度が高い
2. **ループ終了条件**: マージ可能なペアがなくなったら終了
3. **事前トークン単位**: 単語境界を超えたマージを防ぐ
