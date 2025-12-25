# Attentionの数学（Attention Mathematics）

## 学習目標

ソフト辞書を**行列演算**で効率的に実装する方法を学ぶ。これがAttentionの基本形。

## 主要概念

### 1. Q, K, V の定義

```python
# Key: 映画のジャンル特性 (7本, 3次元)
K = torch.tensor([
    [8, 2, 3],  # アクション重視
    [3, 9, 1],  # ドラマ重視
    ...
], dtype=torch.float32)

# Value: ユーザーの評価 (7本, 1次元)
V = torch.tensor([
    [85], [70], [60], ...
], dtype=torch.float32)

# Query: 新しい映画 (3本, 3次元)
Q = torch.tensor([
    [6, 4, 5],  # バランスの取れたアクション寄り
    [2, 8, 3],  # ドラマ重視
    [4, 3, 7],  # コメディ寄り
], dtype=torch.float32)
```

### 2. Attention関数

```python
def attention(Q, K, V):
    similarity = torch.matmul(Q, K.t())     # QK^T を計算
    weights = F.softmax(similarity, dim=1)  # ソフトマックス関数
    output = torch.matmul(weights, V)       # 重み付き和
    return output, weights
```

### 3. 計算の流れ

```
Q: (3, 3)  ←  3つのクエリ、各3次元
K: (7, 3)  ←  7つのキー、各3次元
V: (7, 1)  ←  7つのバリュー、各1次元

Step 1: QK^T
  (3, 3) × (3, 7) = (3, 7)  ← 各クエリと各キーの類似度

Step 2: softmax
  (3, 7) → (3, 7)  ← 各行が確率分布に

Step 3: weights × V
  (3, 7) × (7, 1) = (3, 1)  ← 各クエリに対する出力
```

### 4. 使用例

```python
predicted_ratings, weights = attention(Q, K, V)

for movie, rating in zip(Q, predicted_ratings):
    print(f"映画 {movie.numpy()} の予測評価: {rating.item():.2f}")
```

## ポイント

1. **行列積で一括計算**: forループなしで全クエリを並列処理
2. **K.t()**: Keyの転置でQとの内積を計算
3. **dim=1**: 各クエリに対してソフトマックス（行方向）

## 次のステップ

- スケーリング（次元数による正規化）
- マスク処理（因果的Attention）
- Multi-Head Attention
