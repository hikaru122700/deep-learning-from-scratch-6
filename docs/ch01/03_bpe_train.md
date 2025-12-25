# BPE学習（BPE Training）

## 学習目標

**Byte Pair Encoding（BPE）** のマージルール学習アルゴリズムを理解する。

## 主要概念

### 1. BPEの基本アイデア

1. テキストをバイト列（0-255）に変換
2. 最も頻出する隣接ペアを見つける
3. そのペアを新しいトークンにマージ
4. 目標の語彙サイズになるまで繰り返す

### 2. 隣接ペアのカウント

```python
from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# 使用例
ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
# {(1, 2): 2, (2, 3): 1, (3, 1): 1}
```

- `zip(ids, ids[1:])` で隣接ペアを生成
- 各ペアの出現回数をカウント

### 3. マージ処理

```python
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2  # ペアをスキップ
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

# 使用例
ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
# [4, 3, 4]
```

### 4. BPE学習関数

```python
def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))  # バイト列に変換
    num_merges = vocab_size - 256     # マージ回数
    merge_rules = {}

    for step in range(num_merges):
        counts = count_pairs(ids)
        best_pair = max(counts, key=counts.get)  # 最頻出ペア
        new_id = 256 + step                       # 新しいトークンID
        merge_rules[best_pair] = new_id
        ids = merge(ids, best_pair, new_id)

    return merge_rules
```

### 5. 使用例

```python
text = "Hello world! This is BPE training."
merge_rules = train_bpe(text, vocab_size=260)
# {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}
```

## アルゴリズムの流れ

```
初期状態: [72, 101, 108, 108, 111, ...]  (バイト列)
         ↓
ステップ1: (105, 115) → 256  # "is" をマージ
         ↓
ステップ2: (256, 32) → 257   # "is " をマージ
         ↓
         ...
```

## ポイント

1. **語彙サイズの計算**: `num_merges = vocab_size - 256`
   - 256は初期語彙（バイト値0-255）
2. **新しいトークンID**: 256から順番に割り当て
3. **マージルールの順序**: 学習時の順序が重要（後のエンコードで使用）

## 計算量

- 各ステップで全ペアをカウント → O(n)
- 全ペアに対してマージ → O(n)
- 合計: O(num_merges × n)

→ 大規模テキストでは最適化が必要（後の章で学習）
