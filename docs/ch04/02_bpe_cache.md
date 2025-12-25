# BPEキャッシュ最適化

## 学習目標

**キャッシュ**を使ってBPE学習をさらに高速化する方法を理解する。

## 主要概念

### 1. 問題点

毎ステップで全ID列のペア頻度を再計算するのは無駄：
- マージの影響を受けるのは best_pair を含むID列のみ
- 影響を受けないID列は再計算不要

### 2. キャッシュ戦略

**pair_to_ids**: どのペアがどのID列に含まれるかを記録

```python
pair_to_ids = defaultdict(set)  # キャッシュ

# 初期化時にキャッシュを構築
for ids, count in ids_counts.items():
    for pair in zip(ids, ids[1:]):
        pair_to_ids[pair].add(ids)
```

### 3. 効率的な更新

```python
for step in range(num_merges):
    # 最頻出ペアを選択
    best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))
    new_id = 256 + step
    merge_rules[best_pair] = new_id

    # best_pairを含むID列のみを更新
    affected_ids = pair_to_ids[best_pair]
    del pair_to_ids[best_pair]

    for ids in affected_ids:
        ids_count = ids_counts[tuple(ids)]
        new_ids = merge(ids, best_pair, new_id)

        del ids_counts[tuple(ids)]
        ids_counts[tuple(new_ids)] = ids_count

        # 古いペア頻度を減少
        old_counts = count_pairs(ids)
        for pair, count in old_counts.items():
            pair_counts[pair] -= count * ids_count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            pair_to_ids[pair].discard(tuple(ids))

        # 新しいペア頻度を増加
        new_counts = count_pairs(new_ids)
        for pair, count in new_counts.items():
            pair_counts[pair] += count * ids_count
            pair_to_ids[pair].add(tuple(new_ids))
```

## 計算量の改善

| 項目 | 最適化前 | 最適化後 |
|------|---------|---------|
| 各ステップ | O(全ID列) | O(影響を受けるID列) |
| 全体 | O(num_merges × n) | O(num_merges × k) |

k は各ステップで影響を受けるID列の数（通常 n より大幅に小さい）

## ポイント

1. **差分更新**: 変更があった部分のみ更新
2. **キャッシュの維持**: pair_to_ids を常に最新に保つ
3. **メモリとのトレードオフ**: キャッシュ用のメモリが必要
