# BPE最適化（基本）

## 学習目標

BPE学習の**計算効率を改善**する最初のステップを理解する。

## 主要概念

### 1. 問題点

元のBPE実装では、各ステップで：
- 全テキストのペア頻度をカウント → O(n)
- 最頻出ペアを選択
- 全テキストでマージ → O(n)

これを num_merges 回繰り返すと O(num_merges × n) の計算量。

### 2. 最適化アイデア

**事前トークンの重み付きカウント**：

```python
# 事前トークン化で同じ文字列の出現回数をカウント
pretoken_counts = defaultdict(int)
for text in texts:
    for pretoken in pretokenize_iter(text):
        pretoken_counts[pretoken] += 1

# 事前トークンをID列に変換（出現回数付き）
ids_counts = {
    tuple(pretoken.encode("utf-8")): count
    for pretoken, count in pretoken_counts.items()
}
```

### 3. 重み付きペアカウント

```python
def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight  # 出現回数を重みとして使用
    return counts
```

### 4. 最適化されたBPE学習

```python
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 事前トークン化と重み付け
    pretoken_counts = defaultdict(int)
    for text in texts:
        for pretoken in pretokenize_iter(text):
            pretoken_counts[pretoken] += 1

    ids_counts = {tuple(pretoken.encode("utf-8")): count
                  for pretoken, count in pretoken_counts.items()}

    for step in range(num_merges):
        # 重み付きペア頻度を集計
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)

        # 最頻出ペアを選択（タイブレークあり）
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))

        # マージ...
```

## 改善点

| 項目 | 改善前 | 改善後 |
|------|--------|--------|
| カウント対象 | 全テキスト | ユニークな事前トークン |
| 重複処理 | 毎回カウント | 重みで一括 |

## ポイント

1. **事前トークンの重複除去**: 同じ文字列は1回だけ処理
2. **重み付き**: 出現回数を乗算で反映
3. **タイブレーク**: 同じ頻度のペアは (pair[0], pair[1]) で決定的に選択
