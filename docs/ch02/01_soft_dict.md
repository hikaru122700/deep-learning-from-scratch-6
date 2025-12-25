# ソフト辞書（Soft Dictionary）

## 学習目標

Attentionの直感的な理解のため、**ソフト辞書**の概念を学ぶ。通常の辞書と異なり、類似度に基づいて重み付き平均を返す。

## 主要概念

### 1. 通常の辞書 vs ソフト辞書

| 種類 | 検索方法 | 返り値 |
|------|---------|--------|
| 通常の辞書 | 完全一致 | 単一の値 |
| ソフト辞書 | 類似度ベース | 重み付き平均 |

### 2. ソフト辞書の実装

```python
def soft_dictionary(query, dictionary):
    # 類似度を計算（内積）
    similarity = []
    for key in dictionary:
        s = np.dot(query, key)
        similarity.append(s)

    # ソフトマックスで重みに変換
    exp_similarity = np.exp(similarity)
    weights = exp_similarity / np.sum(exp_similarity)

    # 重み付き和
    result = 0
    for weight, value in zip(weights, dictionary.values()):
        result += weight * value

    return result, weights
```

### 3. 使用例：映画評価の予測

```python
# キー: (アクション性, ドラマ性, コメディ性)
# バリュー: ユーザーの評価点
movie_preferences = {
    (8, 2, 3): 85,   # アクション重視
    (3, 9, 1): 70,   # ドラマ重視
    (1, 2, 9): 60,   # コメディ重視
    (5, 5, 5): 75,   # バランス型
}

# 新しい映画のクエリ
new_movie = (6, 4, 5)

predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)
# → 類似した映画の評価を重み付けして予測
```

## Attentionとの関係

| ソフト辞書 | Attention |
|-----------|-----------|
| Query | Query (Q) |
| Key | Key (K) |
| Value | Value (V) |
| 類似度 | Attention Score |
| 重み | Attention Weight |
| 重み付き和 | Attention Output |

## ポイント

1. **類似度計算**: 内積で計算（高次元でも効率的）
2. **ソフトマックス**: 類似度を確率分布に変換
3. **重み付き和**: すべてのValueを考慮した出力

## Attentionへの橋渡し

このソフト辞書の概念を行列演算で効率化すると、Attentionメカニズムになる。
